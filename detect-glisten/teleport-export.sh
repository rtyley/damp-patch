#!/bin/bash

curl --silent "https://www.teleport.io/api/v1/frame-query/$TELEPORT_FEED_ID?starttime=2023-10-16T00:00:00Z&endtime=2023-10-21T22:00:00Z&apikey=TELEPORT_API_KEY" | jq -r '.Frames[]' | xargs -P 8 -I TIME curl --silent -o TIME.jpg "https://www.teleport.io/api/v1/frame-get?feedid=$TELEPORT_FEED_ID&sizecode=4320p&apikey=$TELEPORT_API_KEY&frametime=TIME"