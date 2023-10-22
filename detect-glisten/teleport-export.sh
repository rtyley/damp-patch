#!/bin/bash

echo "TELEPORT_FEED_ID='$TELEPORT_FEED_ID'"

frames_file=$(mktemp -t teleport-export-frames)

curl --silent "https://www.teleport.io/api/v1/frame-query/$TELEPORT_FEED_ID?starttime=2023-10-16T00:00:00Z&endtime=2023-10-21T22:00:00Z&apikey=$TELEPORT_API_KEY" | jq -r '.Frames[]' > "$frames_file"

echo "Num Frames = $(wc -l $frames_file)"

xargs -P 8 -I TIME curl --silent -o TIME.jpg "https://www.teleport.io/api/v1/frame-get?feedid=$TELEPORT_FEED_ID&sizecode=4320p&apikey=$TELEPORT_API_KEY&frametime=TIME" < "$frames_file"

echo "Export complete!"
