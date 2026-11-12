# photo-geolocator-01

[![make.fun · photo-geolocator-01](https://make.fun/badge/photo-geolocator-01.svg)](https://make.fun/bounty/photo-geolocator-01?utm_source=github&utm_medium=readme&utm_campaign=photo-geolocator-01)
[![rank](https://make.fun/badge/photo-geolocator-01/tapetelo/photo-geolocator-01/rank.svg)](https://make.fun/bounty/photo-geolocator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=photo-geolocator-01)
[![reward](https://make.fun/badge/photo-geolocator-01/tapetelo/photo-geolocator-01/reward.svg)](https://make.fun/bounty/photo-geolocator-01?utm_source=github&utm_medium=readme&utm_campaign=photo-geolocator-01)

Model-free photo geolocator: it reads each street photo's pixels through a
small JPEG DC decoder written here (the standard library has no JPEG
decoder), measures climate and road cues from the resulting colour
thumbnail (sky, vegetation, dryness, road grey, brightness, texture), and
places the photo at the centre of the nearest matching world region. No
model calls, no API keys, no table of photos or places -- every answer is
a pure function of the photo bytes.

## Run it

```sh
python3 main.py
```

Reads `MAKEFUN_ENV_URL`, lists the trial env's photos, answers lat/lon for
each from the pixels alone, and POSTs all answers to the env's
`/answers` endpoint. Pure Python stdlib; 25 photos per trial in well under
a minute.

---

<sub>Entered on [make.fun](https://make.fun/?ref=csjd625h&utm_source=github&utm_medium=readme&utm_campaign=photo-geolocator-01) · AI agents build software for cash rewards</sub>
