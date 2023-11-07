import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Tuple

import aiofiles
import aiohttp
import cv2
import matplotlib.pyplot as plt
import numpy as np
import csv

from credentials import teleport_api_key, teleport_feed_id

# rom segment_anything_hq import SamPredictor, sam_model_registry

# model_type = "vit_h"  # "vit_l/vit_b/vit_h/vit_tiny"
# sam_checkpoint = "/Users/Roberto_Tyley/Downloads/sam_hq_vit_h.pth"
# sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

print(os.environ['HOME'])

teleport_folder = '/tmp/teleport-export'

suffix_for_original_image = '-original.jpg'
suffix_for_mask_image = '-mask.png'

print(f'teleport_feed_id={teleport_feed_id}')

known_dampness_scale_samples: list[datetime] = [datetime.fromisoformat(date_string) for date_string in [
  "2023-10-17T13:02:33Z",  # Scored high (0.078) in the past, but does not look damp to me?! Is it widespread mild damp?
  # "2023-10-19T17:01:24Z",  # A bit of damp, on the left
  "2023-10-19T14:13:32Z",  # Possibly wetter than 2023-10-19T17:01:24Z, definitely wetter than 2023-10-17T13:02:33Z
  "2023-10-29T19:34:23Z",  # Local peak, probably wetter than 2023-10-19T14:13:32Z, but comparable
  "2023-10-18T21:25:24Z",  # Wetter than 2023-10-19T14:13:32Z, less wet than 2023-10-16T20:35:33Z
  "2023-10-16T20:35:33Z",  # Pretty wet!
  "2023-10-19T19:07:34Z"  # Very wet - maybe the most wet!?
]]


async def download_frame_list(session) -> list[datetime]:
  now = datetime.now(timezone.utc)
  url = f'https://www.teleport.io/api/v1/frame-query/{teleport_feed_id}?starttime=2023-10-15T15:00:00Z&endtime={iso_format(now)}&apikey={teleport_api_key}'
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
    full_frame_list = await download_missing_frames(currently_downloaded_frames, session)
    # full_frame_list = sorted(currently_downloaded_frames)
    print(f'Num full_frame_list = {len(full_frame_list)}')

    frame_time = full_frame_list[0]
    image = original_image_for(frame_time)
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('on')
    # plt.show()

    find_bright_pixels(image)

    bright_pixel_frames = [find_bright_pixels(original_image_for(frame_time)) for frame_time in full_frame_list]

    loaded_mask = normalise(cv2.cvtColor(cv2.imread("just_damp_never_bright.edited.png"), cv2.COLOR_BGR2GRAY))
    test_mask(lambda dt: (bright_pixel_frames[full_frame_list.index(dt)] * loaded_mask).sum())
    explore_frame_scoring(datetime.fromisoformat("2023-10-17T13:02:33Z"), loaded_mask)

    dump_frames_scores_csv(full_frame_list, bright_pixel_frames, loaded_mask, "edited_mask")

    normalisedSumOfBrights = normalise(sum(bright_pixel_frames))

    # stepper = 10
    # for b in range(0, stepper):
    #     base = b / stepper
    #     dump_normalised_image(keep_only_within_range(normalisedSumOfBrights, base, base + (1 / stepper)), f"bpf_stepper_{base}.png")

    frame_indices = range(len(full_frame_list))

    constrained = keep_only_within_range(normalisedSumOfBrights, 0.2, 0.7)
    constrained_threshold = 5000
    for index in frame_indices:
      score = ((bright_pixel_frames[index] * loaded_mask) > 0).sum()
      if (score > constrained_threshold):
        print(f'{iso_format(full_frame_list[index])} : score={score}')

    damp_frame_indices = [frame_index for frame_index in frame_indices if
                          ((bright_pixel_frames[frame_index] * loaded_mask) > 0).sum() > constrained_threshold]
    print(f'damp_frame_indices : {len(damp_frame_indices)}')

    dry_frame_indices = list(set(frame_indices) - set(damp_frame_indices))
    # random.shuffle(dry_frame_indices)
    # dry_frame_indices = dry_frame_indices[:len(damp_frame_indices)]

    damp_or_bright = normalised_sum_of_selected_frames(bright_pixel_frames, damp_frame_indices)
    bright = normalised_sum_of_selected_frames(bright_pixel_frames, dry_frame_indices)
    dump_normalised_image(damp_or_bright, f"damp_or_bright.png")
    dump_normalised_image(bright, f"bright.png")

    just_damp_never_bright = np.ma.masked_where(bright > 0, damp_or_bright)
    dump_normalised_image(just_damp_never_bright, f"just_damp_never_bright.png")

    dump_frames_scores_csv(full_frame_list, bright_pixel_frames, just_damp_never_bright,
                           "just_damp_never_bright")

    test_mask(lambda dt: (bright_pixel_frames[full_frame_list.index(dt)] * just_damp_never_bright).sum())
    # for frame_index in full_frame_list:
    #     identify_mask(frame_time)


def explore_frame_scoring(frame_time: datetime, mask):
  print(f"explore_frame_scoring : frame_time={frame_time}")
  bright_pixel_frame = find_bright_pixels(original_image_for(frame_time))
  hot_pixels = bright_pixel_frame * mask
  dump_normalised_image(hot_pixels, f"hot_pixels.png")


def test_mask(scorer: Callable[[datetime], float]):
  frame_times_with_scores: list[tuple[datetime, float]] = [(dt, scorer(dt)) for dt in known_dampness_scale_samples]
  sorted_by_scorer = sorted(enumerate(frame_times_with_scores), key=lambda entry: entry[1][1])
  for index, (frame_time, score) in sorted_by_scorer:
    print(f"{index}. {iso_format(frame_time)} ({score:.2%})")


def dump_frames_scores_csv(full_frame_list, bright_pixel_frames, mask, name: str):
  frame_indices = range(len(full_frame_list))
  scores_based_on_updated_mask = [(frame * mask).sum() for frame in bright_pixel_frames]
  normalised_scores = normalise(np.array(scores_based_on_updated_mask))

  with open(f'{name}.scores.csv', 'w', newline='') as csvfile:
    fieldnames = ['frame_time', 'score']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

    for frame_index in frame_indices:
      writer.writerow({'frame_time': iso_format(full_frame_list[frame_index]), 'score': normalised_scores[frame_index]})


def normalised_sum_of_selected_frames(all_frames, selected_frame_indices):
  print(f"Num selected_frame_indices={len(selected_frame_indices)}")
  # freq_image = np.add.reduce(all_stack, )
  freq_image = sum([all_frames[frame_index] for frame_index in selected_frame_indices])

  print(f"freq_image.shape={freq_image.shape}")
  return normalise(freq_image)


async def download_missing_frames(currently_downloaded_frames, session):
  latest_frame_list = await download_frame_list(session)
  print(f'Num latest_frame_list = {len(latest_frame_list)}')
  missing_frames = sorted(set(latest_frame_list).difference(set(currently_downloaded_frames)))
  print(f'missing_frames = {missing_frames}')
  tasks = [download_frame(session, dt) for dt in missing_frames]
  await asyncio.gather(*tasks)
  return latest_frame_list


def dump_normalised_image(arr, name: str):
  scaled_image = (normalise(arr) * 255).astype(np.uint8)
  cv2.imwrite(name, scaled_image)


def keep_only_within_range(arr, min_v, max_v):
  return ((arr >= min_v) & (arr < max_v)) * arr


def normalise(arr):
  min_val = np.min(arr)
  max_val = np.max(arr)
  print(f'min_val={min_val} max_val={max_val}')
  return (arr - min_val) / (max_val - min_val)


def find_bright_pixels(image):
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  # cv2.imshow('Grayscale', gray_image)
  threshold = 200
  ret, threshold_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_TOZERO)
  return threshold_image


def original_image_for(frame_time):
  return cv2.cvtColor(cv2.imread(original_image_file_for(frame_time)), cv2.COLOR_BGR2RGB)


# def identify_mask(frame_time: datetime):
#     image = original_image_for(frame_time)
#     damp_coords = [[864, 1183], [432, 3444], [732, 2500], [732, 2000], [434, 1607], [165, 1399], [1155, 2507],
#                    [2311, 3116], [1879, 3153]]
#     non_damp_coords = [[1630, 1690], [856, 2314], [848, 2159], [875, 2110], [1084, 1900], [1109, 1861], [876, 2036]]
#
#     input_point = np.array(damp_coords + non_damp_coords)
#     input_label = np.array(([1] * len(damp_coords)) + ([0] * len(non_damp_coords)))
#     predictor = SamPredictor(sam)
#     predictor.set_image(image)
#     masks, scores, _ = predictor.predict(point_coords=input_point, point_labels=input_label)
#     print(masks.shape)
#     mask = masks[0]
#     print(f'{iso_format(frame_time)} : {(mask > 0).sum()}')
#     mask_image = (mask * 255).astype(np.uint8)
#     cv2.imwrite(mask_file_for(frame_time), mask_image)


asyncio.get_event_loop().run_until_complete(do_tha_bizness())
