import streamlink

stream_link = "https://www.earthcam.com/usa/newyork/timessquare/?cam=tsrobo1"
streams = streamlink.streams(stream_link)
if streams:
    stream_url = streams['best'].url
    print("Stream URL:", stream_url)
else:
    print("No streams found")