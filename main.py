import argparse

from analyzer import process_video

parser = argparse.ArgumentParser(description='Vehicle speed detector.')
parser.add_argument('video_source', metavar='video', type=str, help='Filepath or stream URL')
parser.add_argument('-v, --visual', dest='visual', action='store_true', help='Visual mode with video output')
parser.add_argument('-s, --speed', dest='speed_limit', type=int, help='Speed limit', required=True)
parser.add_argument('-z, --zone', dest='zone_length', type=int, help='Zone length', required=True)

args = parser.parse_args()
visual_mode = args.visual
speed_limit = args.speed_limit
zone_length = args.zone_length

process_video(
    source=args.video_source,
    show_output=visual_mode,
    speed_limit=speed_limit,
    zone_length=zone_length
)
