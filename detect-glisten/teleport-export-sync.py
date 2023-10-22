import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path

import aiofiles
import aiohttp

print(os.environ['HOME'])

teleport_feed_id = os.environ['TELEPORT_FEED_ID']
teleport_api_key = os.environ['TELEPORT_API_KEY']
teleport_folder = '/tmp/teleport-export'

print(f'teleport_feed_id={teleport_feed_id}')


async def download_frame_list(session) -> list[datetime]:
    now = datetime.now(timezone.utc)
    url = f'https://www.teleport.io/api/v1/frame-query/{teleport_feed_id}?starttime=2023-10-16T00:00:00Z&endtime={iso_format(now)}&apikey={teleport_api_key}'
    print(url)
    async with session.get(url) as response:
        return [datetime.fromisoformat(date_string) for date_string in (await response.json())['Frames']]


def original_image_file_for(dt: datetime) -> str:
    return f'{teleport_folder}/{iso_format(dt)}.jpg'


def iso_format(dt):
    return dt.isoformat().replace("+00:00", "Z")


async def download_frame(session, dt: datetime):
    url = f'https://www.teleport.io/api/v1/frame-get?feedid={teleport_feed_id}&sizecode=4320p&apikey={teleport_api_key}&frametime={iso_format(dt)}'
    async with session.get(url) as resp:
        if resp.status == 200:
            file_path = original_image_file_for(dt)
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(await resp.read())


def list_currently_downloaded_frames() -> list[datetime]:
    return [datetime.fromisoformat(file_name.name.strip(".jpg")) for file_name in Path(teleport_folder).glob('*.jpg')]


async def do_tha_bizness():
    os.makedirs(teleport_folder, exist_ok=True)
    currently_downloaded_frames = list_currently_downloaded_frames()
    print(f'currently_downloaded_frames={currently_downloaded_frames}')
    async with aiohttp.ClientSession() as session:
        full_frame_list = await download_frame_list(session)
        print(f'full_frame_list = {full_frame_list}')
        missing_frames = sorted(set(full_frame_list).difference(set(currently_downloaded_frames)))
        print(f'missing_frames = {missing_frames}')
        tasks = [download_frame(session, dt) for dt in missing_frames]
        await asyncio.gather(*tasks)


asyncio.get_event_loop().run_until_complete(do_tha_bizness())
