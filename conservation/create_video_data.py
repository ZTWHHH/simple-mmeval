import json
import re
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MEDIA_DIR = os.path.join(SCRIPT_DIR, "media")
CONTROL_DIR = os.path.join(SCRIPT_DIR, "../../simple-mmeval/Conservation_control")
INPUT_FILE = os.path.join(SCRIPT_DIR, "v1_7frames.json")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "v1_video_description.json")

def build_video_lookup():
    """Build a case-insensitive lookup from normalized video name to actual filename.

    When both lowercase and Title case versions exist (e.g. number_exp1.mp4 and
    Number_Exp1.mp4), prefer lowercase for exp videos. Control videos only have
    Title case versions (Length_Control_1.mp4, Liquid_Control_1.MOV).
    """
    lookup = {}

    for fname in sorted(os.listdir(MEDIA_DIR)):
        if fname.lower().endswith(".mp4"):
            key = fname.lower().replace(".mp4", "")
            if key in lookup and fname[0].islower():
                lookup[key] = fname
            elif key not in lookup:
                lookup[key] = fname

    if os.path.isdir(CONTROL_DIR):
        for fname in sorted(os.listdir(CONTROL_DIR)):
            if fname.upper().endswith(".MOV"):
                key = fname.lower().replace(".mov", "")
                if key not in lookup:
                    lookup[key] = fname

    return lookup


def extract_video_id(media_list):
    """Extract video identifier from the first media filename.

    e.g. 'u_8f_number_exp1_1.png' -> 'number_exp1'
         'h_8f_length_control12_3.png' -> 'length_control12'
    """
    first = media_list[0]
    without_frame = first.rsplit("_", 1)[0]  # strip '_1.png' -> 'u_8f_number_exp1'
    m = re.match(r"^[hsu]_8f_(.+)$", without_frame)
    if m:
        return m.group(1)
    return without_frame


def video_id_to_lookup_key(vid_id):
    """Convert a video ID to possible lookup keys.

    'number_exp1' -> ['number_exp1']
    'length_control12' -> ['length_control12', 'length_control_12']
    """
    keys = [vid_id.lower()]
    m = re.match(r"^(.+?[a-z])(\d+)$", vid_id)
    if m:
        base = m.group(1)
        num = m.group(2)
        keys.append(f"{base}_{num}".lower())
    return keys


def main():
    video_lookup = build_video_lookup()
    print(f"Video lookup: {len(video_lookup)} entries")

    with open(INPUT_FILE, "r") as f:
        data = json.load(f)
    print(f"Input entries: {len(data)}")

    seen_videos = set()
    new_data = []
    missing = []

    for item in data:
        vid_id = extract_video_id(item["media"])

        if vid_id in seen_videos:
            continue
        seen_videos.add(vid_id)

        video_file = None
        for key in video_id_to_lookup_key(vid_id):
            if key in video_lookup:
                video_file = video_lookup[key]
                break

        if video_file is None:
            missing.append(vid_id)
            continue

        new_item = {
            "question": "<video> Please provide a detailed description for the video.\n",
            "media": [video_file],
            "concept_type": item.get("concept_type"),
            "type": item.get("type"),
            "stage_lvl": item.get("stage_lvl"),
            "batch_id": item.get("batch_id"),
            "id": item.get("id"),
            "messages": [
                {
                    "role": "user",
                    "prompt": "<video> Please provide a detailed description for the video.",
                }
            ],
        }
        new_data.append(new_item)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(new_data, f, indent=4, ensure_ascii=False)

    print(f"Output entries: {len(new_data)}")
    if missing:
        print(f"WARNING: {len(missing)} videos not found: {missing}")
    else:
        print("All videos matched successfully.")


if __name__ == "__main__":
    main()
