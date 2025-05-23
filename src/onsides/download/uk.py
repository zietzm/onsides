import asyncio
import logging
import re
from enum import Enum

import httpx
from aiolimiter import AsyncLimiter
from bs4 import BeautifulSoup, Tag
from bs4.element import NavigableString
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlmodel import col, select

from onsides.db import DrugLabel, DrugLabelSource
from onsides.state import State

logger = logging.getLogger(__name__)
limiter = AsyncLimiter(2, 5)

ROOT_URL = "https://www.medicines.org.uk"


async def download_uk(state: State) -> None:
    async with httpx.AsyncClient() as client:
        logger.debug("UK: Downloading remaining drug labels")
        await scrape_products(client, state)
        logger.info("UK: Finished downloading")


async def scrape_products(
    client: httpx.AsyncClient,
    state: State,
) -> None:
    logger.info("UK: Getting the list of drug prefixes")
    drug_prefix_urls = await get_page_links(client)
    if isinstance(drug_prefix_urls, AcquireLabelError):
        raise RuntimeError(f"UK: Couldn't get drug pages: {drug_prefix_urls}")

    logger.info("UK: Gathering all drgus on the EMC website")
    tasks = list()
    async_session = state.get_async_session()
    async with async_session() as session:
        for prefix_url in drug_prefix_urls:
            task = asyncio.create_task(
                process_prefix(prefix_url, client, session, state)
            )
            tasks.append(task)

        logger.info("UK: Awaiting results")
        await asyncio.gather(*tasks)
        logger.info("UK: Finished gathering drug stubs")

    await download_all_labels(state, async_session, client)

    return None


class AcquireLabelError(Enum):
    RATE_LIMIT = 1
    OTHER_HTTP_ERROR = 2
    NO_BROWSE_MENU = 3
    NO_COUNT = 4
    NO_CLINICAL_PARTICULARS = 5


class DownloadError(Enum):
    RATE_LIMIT = 1
    OTHER_HTTP_ERROR = 2


async def get_helper(url: str, client: httpx.AsyncClient) -> str | DownloadError:
    async with limiter:
        try:
            response = await client.get(url, timeout=10.0)
        except httpx.HTTPError as exc:
            logger.error(f"UK: HTTP Exception for {exc.request.url} - {exc}")
            return DownloadError.OTHER_HTTP_ERROR

    try:
        response.raise_for_status()
    except httpx.HTTPError:
        logger.error(f"EU: Error status: {response.status_code}")
        if response.status_code == 429:
            return DownloadError.RATE_LIMIT
        return DownloadError.OTHER_HTTP_ERROR

    return response.text


async def get_page_links(client: httpx.AsyncClient) -> list[str] | AcquireLabelError:
    result = await get_helper(f"{ROOT_URL}/emc/browse-medicines", client)
    if isinstance(result, DownloadError):
        return AcquireLabelError(result.value)

    soup = BeautifulSoup(result, "html.parser")
    section = soup.find("div", {"class": "browse-menu"})
    if section is None or isinstance(section, NavigableString):
        logger.error("UK: Couldn't find the browse menu")
        return AcquireLabelError.NO_BROWSE_MENU

    assert isinstance(section, Tag)
    links = section.find_all("a", {"class": "emc-link"}, href=True)
    reg = r"^/emc/browse-medicines/.+$"
    results = list()
    for link in links:
        match = re.match(reg, link["href"])  # type: ignore
        if match is not None:
            results.append(match.group(0))
    return results


async def process_prefix(
    prefix_url: str,
    client: httpx.AsyncClient,
    async_session: AsyncSession,
    state: State,
) -> AcquireLabelError | None:
    prefix = prefix_url.replace("/emc/browse-medicines/", "")
    first_page = await get_helper(f"{ROOT_URL}{prefix_url}?offset=0&limit=200", client)
    if isinstance(first_page, DownloadError):
        if first_page == DownloadError.RATE_LIMIT:
            return await process_prefix(prefix_url, client, async_session, state)
        return AcquireLabelError(first_page.value)

    await add_drug_stubs_to_db(first_page, async_session)

    n_drugs = count_drugs_under_prefix(first_page)
    if isinstance(n_drugs, AcquireLabelError):
        return n_drugs
    logger.info(f"UK: Found {n_drugs} drugs in prefix {prefix}")
    n_additional_pages = (n_drugs - 1) // 200  # 200 -> 0, 201 -> 1
    if n_additional_pages == 0:
        return None

    task = state.add_task(
        f"uk-{prefix}",
        f"UK: Gathering pages for prefix: {prefix}...",
        total=n_additional_pages + 1,
    )
    state.progress.update(task, completed=1)
    for page_num in range(n_additional_pages):
        offset = 201 + 200 * page_num
        page = await get_helper(
            f"{ROOT_URL}{prefix_url}?offset={offset}&limit=200", client
        )
        if isinstance(page, DownloadError):
            if page == DownloadError.RATE_LIMIT:
                logger.error(f"UK: Rate limit error for {prefix}")
                await asyncio.sleep(1)
                continue
            return AcquireLabelError(page.value)

        await add_drug_stubs_to_db(page, async_session)
        state.progress.update(task, advance=1)

    state.progress.update(task, visible=False)
    return None


def count_drugs_under_prefix(first_page: str) -> int | AcquireLabelError:
    soup = BeautifulSoup(first_page, "html.parser")
    section = soup.find(
        "span", {"class": "latest-updates-results-header-summary-total"}
    )
    if section is None:
        return AcquireLabelError.NO_COUNT

    match = re.search(r"([0-9]+) results found", section.text)
    if match is None:
        return AcquireLabelError.NO_COUNT

    return int(match.group(1))


async def add_drug_stubs_to_db(page: str, session: AsyncSession) -> None:
    soup = BeautifulSoup(page, "html.parser")
    product_elements = soup.find_all("div", {"class": "search-results-product"})

    for product in product_elements:
        if not isinstance(product, Tag):
            logger.warning(f"UK: Unknown type {product}")
            continue

        name_element = product.find(
            "a",
            {"class": "search-results-product-info-title-link emc-link"},
            href=True,
        )
        if name_element is None:
            logger.warning(f"UK: No name element: {product}")
            continue

        link = str(name_element["href"])  # type: ignore
        if link.endswith("/pil"):
            # These links don't seem to work.
            source_id = link.replace("/pil", "").replace("/emc/product/", "")
            full_url = None
        else:
            source_id = link.replace("/smpc", "").replace("/emc/product/", "")
            full_url = f"{ROOT_URL}{link}"

        # Look for an existing DrugLabel
        query = (
            select(DrugLabel)
            .where(DrugLabel.source == DrugLabelSource.UK)
            .where(DrugLabel.source_id == source_id)
        )
        result = await session.execute(query)
        label = result.scalar_one_or_none()

        # If not found, create and add a new label
        if label is None:
            label = DrugLabel(
                source=DrugLabelSource.UK,
                source_id=source_id,
                source_name=name_element.text,
                label_url=full_url,
                raw_text=None,
            )
            session.add(label)
            await session.commit()
    return None


async def get_drugs_from_page(
    page: str,
    client: httpx.AsyncClient,
    async_session: async_sessionmaker[AsyncSession],
) -> list[DrugLabel]:
    soup = BeautifulSoup(page, "html.parser")
    results = []
    product_elements = soup.find_all("div", {"class": "search-results-product"})

    for product in product_elements:
        if not isinstance(product, Tag):
            logger.warning(f"UK: Unknown type {product}")
            continue

        name_element = product.find(
            "a", {"class": "search-results-product-info-title-link emc-link"}, href=True
        )
        if name_element is None:
            logger.warning(f"UK: No name element: {product}")
            continue

        link = str(name_element["href"])  # type: ignore
        if link.endswith("/pil"):
            source_id = link.replace("/pil", "").replace("/emc/product/", "")
            full_url = None
        else:
            source_id = link.replace("/smpc", "").replace("/emc/product/", "")
            full_url = f"{ROOT_URL}{link}"

        async with async_session() as session, session.begin():
            # Look for an existing DrugLabel
            query = (
                select(DrugLabel)
                .where(DrugLabel.source == DrugLabelSource.UK)
                .where(DrugLabel.source_id == source_id)
            )
            result = await session.execute(query)
            label = result.scalar_one_or_none()

            # If not found, create and add a new label
            if label is None:
                label = DrugLabel(
                    source=DrugLabelSource.UK,
                    source_id=source_id,
                    source_name=name_element.text,
                    label_url=full_url,
                    raw_text=None,
                )
                session.add(label)

            # If raw_text is missing, fetch and update it
            if label.raw_text is None and label.label_url is not None:
                raw_text = await get_info_page(label.label_url, client)
                if isinstance(raw_text, AcquireLabelError):
                    logger.error(f"UK: Error fetching info page: {label.label_url}")
                    continue
                label.raw_text = raw_text

            results.append(label)

    logger.debug(f"UK: Returning {len(results)}")
    return results


async def get_info_page(url: str, client: httpx.AsyncClient) -> str | AcquireLabelError:
    page = await get_helper(f"{url}/print", client)
    if isinstance(page, DownloadError):
        return AcquireLabelError(page.value)

    return BeautifulSoup(page, "html.parser").get_text()


async def download_all_labels(
    state: State,
    async_session: async_sessionmaker[AsyncSession],
    client: httpx.AsyncClient,
) -> None:
    query = (
        select(DrugLabel)
        .where(DrugLabel.source == "UK")
        .where(col(DrugLabel.label_url).is_not(None))  # We have the URL
        .where(col(DrugLabel.label_url).like("%smpc%"))  # The URL is usable
        .where(col(DrugLabel.raw_text).is_(None))  # We don't have label text
    )
    async with async_session() as session:
        result = await session.execute(query)
        labels = result.scalars().all()
        logger.info(f"UK: Found {len(labels)} labels with raw text missing")
        task = state.add_task(
            "uk-download-text", "UK: Downloading text...", total=len(labels)
        )

        for label in labels:
            assert label.label_url is not None
            info_page = await get_info_page(label.label_url, client)
            if isinstance(info_page, AcquireLabelError):
                continue

            label.raw_text = info_page
            await session.commit()
            state.progress.update(task, advance=1)

    state.progress.update(task, visible=False)
    return None
