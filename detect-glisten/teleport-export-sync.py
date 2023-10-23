import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import aiofiles
import aiohttp
import cv2

from segment_anything_hq import SamPredictor, sam_model_registry

model_type = "vit_h"  # "vit_l/vit_b/vit_h/vit_tiny"
sam_checkpoint = "/Users/Roberto_Tyley/Downloads/sam_hq_vit_h.pth"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

print(os.environ['HOME'])

teleport_feed_id = os.environ['TELEPORT_FEED_ID']
teleport_api_key = os.environ['TELEPORT_API_KEY']
teleport_folder = '/tmp/teleport-export'

suffix_for_original_image = '-original.jpg'
suffix_for_mask_image = '-mask.png'

print(f'teleport_feed_id={teleport_feed_id}')


async def download_frame_list(session) -> list[datetime]:
    now = datetime.now(timezone.utc)
    url = f'https://www.teleport.io/api/v1/frame-query/{teleport_feed_id}?starttime=2023-10-16T00:00:00Z&endtime={iso_format(now)}&apikey={teleport_api_key}'
    print(url)
    async with session.get(url) as response:
        return [datetime.fromisoformat(date_string) for date_string in (await response.json())['Frames']]


def original_image_file_for(dt: datetime) -> str:
    return f'{teleport_folder}/{iso_format(dt)}{suffix_for_original_image}'


def mask_file_for(dt: datetime) -> str:
    return f'{teleport_folder}/{iso_format(dt)}{suffix_for_mask_image}'


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
    return [datetime.fromisoformat(file_name.name.strip(suffix_for_original_image)) for file_name in
            Path(teleport_folder).glob(f'*{suffix_for_original_image}')]


async def do_tha_bizness():
    os.makedirs(teleport_folder, exist_ok=True)
    currently_downloaded_frames = list_currently_downloaded_frames()
    print(f'Num currently_downloaded_frames={len(currently_downloaded_frames)}')
    async with aiohttp.ClientSession() as session:
        full_frame_list = await download_frame_list(session)
        print(f'Num full_frame_list = {len(full_frame_list)}')
        missing_frames = sorted(set(full_frame_list).difference(set(currently_downloaded_frames)))
        print(f'missing_frames = {missing_frames}')
        tasks = [download_frame(session, dt) for dt in missing_frames]
        await asyncio.gather(*tasks)

        frame_time = full_frame_list[0]
        image = original_image_for(frame_time)
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('on')
        plt.show()

        for frame_time in full_frame_list:
            identify_mask(frame_time)


def original_image_for(frame_time):
    return cv2.cvtColor(cv2.imread(original_image_file_for(frame_time)), cv2.COLOR_BGR2RGB)


def identify_mask(frame_time: datetime):
    image = original_image_for(frame_time)
    input_point = np.array([[732, 2500], [732, 2000], [1630, 1690]])
    input_label = np.array([1, 1, 0])
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(point_coords=input_point, point_labels=input_label)
    print(masks.shape)
    mask = masks[0]
    print(f'{iso_format(frame_time)} : {(mask > 0).sum()}')
    mask_image = (mask * 255).astype(np.uint8)
    cv2.imwrite(mask_file_for(frame_time), mask_image)


asyncio.get_event_loop().run_until_complete(do_tha_bizness())
