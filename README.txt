Android code is based on the OpenCV SDK samples.

arucopage.pdf  -- the markers for the experiments
collectdata.py -- a script to collect data from the Android code: adb logcat | python collectdata.py output.txt
collected-to-rms.py -- output rms and maximum error by z height from raw data collected by collectdata.py