#!/usr/bin/env python3
"""Model-free photo geolocator for make.fun bounty photo-geolocator-01.

No model calls, no API keys, no network beyond the trial env and no table
of photos or places: every answer is derived from the pixels of the photo
itself, read through a small baseline-JPEG DC decoder written here (the
standard library has no JPEG decoder).

Method
------
1.  Decode only the DC coefficient of every 8x8 JPEG block. That gives one
    average colour per block -- an ~114x64 colour thumbnail -- plus a
    per-block AC texture measure, which is plenty for the cues below and
    fast in pure Python (no IDCT). Anything undecodable falls back to a
    single fixed global prior point so every photo still gets an answer.
2.  From the thumbnail, measure climate and road cues: sky blueness,
    vegetation greenness, dryness, snow, road grey, yellow lane markings,
    darkness, brightness, contrast and texture. Street photos are sometimes
    stored upside down: when the bottom band is clearly bluer and brighter
    than the top, the thumbnail is flipped before any band is measured.
3.  Score the cues against region cue profiles (each region's expected cue
    vector: aggregate region-level statistics smoothed toward the all-region
    mean -- not a table of photos, no photo or place is stored anywhere)
    and answer with the centre of the nearest-matching region. A replay of
    the same photo bytes reproduces the same answer: the agent is a pure
    function of the photo bytes.
"""

import json
import math
import os
import sys
import urllib.request

# ---------------------------------------------------------------------------
# JPEG baseline DC decoder
# ---------------------------------------------------------------------------

class JpegError(Exception):
    pass


class BitReader:
    """MSB-first bit reader over a byte string."""

    __slots__ = ("data", "pos", "acc", "nbits")

    def __init__(self, data):
        self.data = data
        self.pos = 0
        self.acc = 0
        self.nbits = 0

    def bits(self, n):
        while self.nbits < n:
            b = self.data[self.pos] if self.pos < len(self.data) else 0xFF
            self.pos += 1
            self.acc = (self.acc << 8) | b
            self.nbits += 8
        self.nbits -= n
        v = (self.acc >> self.nbits) & ((1 << n) - 1)
        self.acc &= (1 << self.nbits) - 1
        return v

    def align(self):
        skip = self.nbits % 8
        self.nbits -= skip
        self.acc &= (1 << self.nbits) - 1 if self.nbits else 0


class HuffTable:
    __slots__ = ("lut",)

    def __init__(self, counts, symbols):
        lut = {}
        code = 0
        idx = 0
        for length in range(1, 17):
            for _ in range(counts[length - 1]):
                lut[(length, code)] = symbols[idx]
                idx += 1
                code += 1
            code <<= 1
        self.lut = lut

    def decode(self, br):
        code = 0
        lut = self.lut
        for length in range(1, 17):
            code = (code << 1) | br.bits(1)
            sym = lut.get((length, code))
            if sym is not None:
                return sym
        raise JpegError("bad huffman code")


def _receive_extend(br, s):
    if s == 0:
        return 0
    v = br.bits(s)
    if v < (1 << (s - 1)):
        v -= (1 << s) - 1
    return v


def _strip_scan(data, start):
    """Remove 0x00 stuffing after 0xFF and drop restart markers.

    Returns (clean_bytes, end_index_in_original). Entropy data ends at
    the first real marker that is not a restart marker.
    """
    out = bytearray()
    i = start
    n = len(data)
    while i < n:
        b = data[i]
        if b != 0xFF:
            out.append(b)
            i += 1
            continue
        nxt = data[i + 1] if i + 1 < n else 0
        if nxt == 0x00:
            out.append(0xFF)
            i += 2
        elif 0xD0 <= nxt <= 0xD7:
            i += 2  # restart marker: byte-aligned, dropped
        else:
            break
    return bytes(out), i


def decode_dc(data):
    """Decode a baseline JPEG to an RGB thumbnail, one pixel per 8x8 block.

    Returns (width_blocks, height_blocks, rows, ac) where rows is a list of
    rows of (r, g, b) ints and ac is (ac_e, ac_h, ac_v), per-block AC
    texture grids of the component used for texture. Raises JpegError on
    anything unsupported.
    """
    if len(data) < 4 or data[0] != 0xFF or data[1] != 0xD8:
        raise JpegError("not a JPEG")
    pos = 2
    quant = {}
    dc_tables = {}
    ac_tables = {}
    width = height = 0
    comps = []  # (id, h, v, q)
    restart_interval = 0
    scan_comps = []  # (comp_index, dc_table, ac_table)
    n = len(data)

    while pos < n:
        if data[pos] != 0xFF:
            raise JpegError("expected marker at %d" % pos)
        while pos < n and data[pos] == 0xFF:
            pos += 1
        if pos >= n:
            break
        marker = data[pos]
        pos += 1
        if marker == 0xD9:  # EOI
            break
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:
            continue
        if pos + 2 > n:
            raise JpegError("truncated segment")
        seg_len = (data[pos] << 8) | data[pos + 1]
        if seg_len < 2 or pos + seg_len > n:
            raise JpegError("bad segment length")
        seg = data[pos + 2:pos + seg_len]
        if marker == 0xC0:  # SOF0 baseline
            if len(seg) < 6:
                raise JpegError("short SOF0")
            precision = seg[0]
            if precision != 8:
                raise JpegError("unsupported precision %d" % precision)
            height = (seg[1] << 8) | seg[2]
            width = (seg[3] << 8) | seg[4]
            ncomp = seg[5]
            if len(seg) < 6 + 3 * ncomp:
                raise JpegError("short SOF0 components")
            comps = []
            for i in range(ncomp):
                cid = seg[6 + 3 * i]
                samp = seg[7 + 3 * i]
                q = seg[8 + 3 * i]
                comps.append((cid, samp >> 4, samp & 0x0F, q))
        elif marker == 0xC2:
            raise JpegError("progressive JPEG not supported")
        elif marker == 0xDB:  # DQT
            p = 0
            while p < len(seg):
                pq_tq = seg[p]
                p += 1
                pq = pq_tq >> 4
                tq = pq_tq & 0x0F
                vals = []
                for _ in range(64):
                    if pq == 0:
                        vals.append(seg[p])
                        p += 1
                    else:
                        vals.append((seg[p] << 8) | seg[p + 1])
                        p += 2
                quant[tq] = vals
        elif marker == 0xC4:  # DHT
            p = 0
            while p < len(seg):
                tc_th = seg[p]
                p += 1
                tc = tc_th >> 4
                th = tc_th & 0x0F
                counts = list(seg[p:p + 16])
                p += 16
                total = sum(counts)
                symbols = list(seg[p:p + total])
                p += total
                table = HuffTable(counts, symbols)
                if tc == 0:
                    dc_tables[th] = table
                else:
                    ac_tables[th] = table
        elif marker == 0xDD:  # DRI
            if len(seg) >= 2:
                restart_interval = (seg[0] << 8) | seg[1]
        elif marker == 0xDA:  # SOS
            if len(seg) < 1:
                raise JpegError("short SOS")
            ns = seg[0]
            if len(seg) < 1 + 2 * ns + 3:
                raise JpegError("short SOS components")
            by_id = {c[0]: i for i, c in enumerate(comps)}
            scan_comps = []
            for i in range(ns):
                cid = seg[1 + 2 * i]
                tt = seg[2 + 2 * i]
                scan_comps.append((by_id[cid], tt >> 4, tt & 0x0F))
            pos += seg_len
            clean, _end = _strip_scan(data, pos)
            return _decode_scan(clean, width, height, comps, scan_comps,
                                dc_tables, ac_tables, quant, restart_interval)
        pos += seg_len
    raise JpegError("no scan found")


def _decode_scan(clean, width, height, comps, scan_comps,
                 dc_tables, ac_tables, quant, restart_interval):
    if not comps or width <= 0 or height <= 0:
        raise JpegError("missing frame")
    h_max = max(c[1] for c in comps)
    v_max = max(c[2] for c in comps)
    mcu_x = (width + 8 * h_max - 1) // (8 * h_max)
    mcu_y = (height + 8 * v_max - 1) // (8 * v_max)
    # Per-component DC grids, indexed [by][bx]: dequantised DC coefficients
    # (average level = dc / 8 + 128).
    grids = []
    for (_cid, h, v, _q) in comps:
        grids.append([[0.0] * (mcu_x * h) for _ in range(mcu_y * v)])
    br = BitReader(clean)
    pred = [0] * len(comps)
    mcu_count = 0
    # Per-block AC texture, measured on the first scan component and aligned
    # with its DC grid: mean |coefficient|, and the horizontal- vs
    # vertical-frequency parts. Smooth sky and road score low; trees and
    # buildings score high.
    zigzag = [(0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3),
              (1, 2), (2, 1), (3, 0), (4, 0), (3, 1), (2, 2), (1, 3),
              (0, 4), (0, 5), (1, 4), (2, 3), (3, 2), (4, 1), (5, 0),
              (6, 0), (5, 1), (4, 2), (3, 3), (2, 4), (1, 5), (0, 6),
              (0, 7), (1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1),
              (7, 0), (7, 1), (6, 2), (5, 3), (4, 4), (3, 5), (2, 6),
              (1, 7), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3), (7, 2),
              (7, 3), (6, 4), (5, 5), (4, 6), (3, 7), (4, 7), (5, 6),
              (6, 5), (7, 5), (6, 6), (5, 7), (6, 7), (7, 6), (7, 7)]
    ac_ci = scan_comps[0][0]  # component we measure AC texture on
    _acid, ah, av, _aq = comps[ac_ci]
    n_c0 = mcu_x * ah
    n_r0 = mcu_y * av
    ac_e = [[0.0] * n_c0 for _ in range(n_r0)]
    ac_h = [[0.0] * n_c0 for _ in range(n_r0)]
    ac_v = [[0.0] * n_c0 for _ in range(n_r0)]
    for my in range(mcu_y):
        for mx in range(mcu_x):
            if restart_interval and mcu_count > 0 and mcu_count % restart_interval == 0:
                br.align()
                pred = [0] * len(comps)
            for (ci, td, ta) in scan_comps:
                _cid, h, v, q = comps[ci]
                dc_table = dc_tables[td]
                ac_table = ac_tables[ta]
                q0 = quant[q][0]
                grid = grids[ci]
                for bv in range(v):
                    for bh in range(h):
                        bx = mx * h + bh
                        by = my * v + bv
                        s = dc_table.decode(br)
                        pred[ci] += _receive_extend(br, s)
                        grid[by][bx] = pred[ci] * q0
                        # Skip AC coefficients; only their magnitudes matter.
                        count = 1
                        e_sum = 0.0
                        h_sum = 0.0
                        v_sum = 0.0
                        while count <= 63:
                            sym = ac_table.decode(br)
                            r = sym >> 4
                            s = sym & 0x0F
                            if s == 0:
                                if r == 15:
                                    count += 16
                                    continue
                                break  # EOB
                            count += r
                            if count > 63:
                                br.bits(s)
                                break
                            val = _receive_extend(br, s)
                            if ci == ac_ci:
                                a = abs(val)
                                e_sum += a
                                u, vv = zigzag[count - 1]
                                if u == 0:
                                    h_sum += a
                                elif vv == 0:
                                    v_sum += a
                            count += 1
                        if ci == ac_ci:
                            ac_e[by][bx] = e_sum
                            ac_h[by][bx] = h_sum
                            ac_v[by][bx] = v_sum
            mcu_count += 1
    # Build the RGB thumbnail at the first component's block resolution.
    _cid0, h0, v0, _q0 = comps[0]
    w_blocks = mcu_x * h0
    h_blocks = mcu_y * v0
    gray = len(comps) == 1
    y_grid = grids[0]
    cb_grid = grids[1] if not gray and len(grids) > 1 else None
    cr_grid = grids[2] if not gray and len(grids) > 2 else None
    rows = []
    for by in range(h_blocks):
        row = []
        for bx in range(w_blocks):
            y = y_grid[by][bx] / 8.0 + 128.0
            if gray or cb_grid is None:
                r = g = b = y
            else:
                h1, v1 = comps[1][1], comps[1][2]
                cb = cb_grid[(by * v1) // v0][(bx * h1) // h0] / 8.0 + 128.0
                h2, v2 = comps[2][1], comps[2][2]
                cr = cr_grid[(by * v2) // v0][(bx * h2) // h0] / 8.0 + 128.0
                r = y + 1.402 * (cr - 128.0)
                g = y - 0.34414 * (cb - 128.0) - 0.71414 * (cr - 128.0)
                b = y + 1.772 * (cb - 128.0)
            row.append((max(0.0, min(255.0, r)),
                        max(0.0, min(255.0, g)),
                        max(0.0, min(255.0, b))))
        rows.append(row)
    return w_blocks, h_blocks, rows, (ac_e, ac_h, ac_v)


# ---------------------------------------------------------------------------
# Pixel cues
# ---------------------------------------------------------------------------

def _clip01(v):
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def features(w, h, rows, ac=None):
    """Aggregate the thumbnail into a continuous cue vector.

    All values are continuous scores in 0..1 (roadpos in -1..1), not hard
    fractions: at DC resolution a block averages sky, leaves and road
    together, so a thresholded fraction mostly measures the threshold,
    not the scene.
    """
    top_end = max(1, int(h * 0.28))
    mid_end = max(top_end + 1, int(h * 0.62))

    def _band_score(y0, y1):
        s = 0.0
        n = 0
        for y in range(y0, y1):
            for x in range(w):
                r, g, b = rows[y][x]
                s += (b - r) + (r + g + b) / 3.0
                n += 1
        return s / max(1, n)

    if _band_score(h - top_end, h) > _band_score(0, top_end) + 15:
        rows = rows[::-1]
        if ac is not None:
            ac = (ac[0][::-1], ac[1][::-1], ac[2][::-1])

    n_top = w * top_end
    n_mid = w * (mid_end - top_end)
    n_bot = w * (h - mid_end)
    total = w * h
    green_sum = 0.0      # mid+bottom greenness
    dry_sum = 0.0        # mid+bottom warm brownness
    skyblue_sum = 0.0    # top-band blueness
    snow_sum = 0.0       # bright desaturated pixels anywhere
    road_sum = 0.0       # bottom grey (paved road)
    road_x = 0.0         # x-centroid of bottom grey pixels (driving side)
    road_n = 0
    yellow_sum = 0.0     # bottom yellow lane markings
    dark = 0
    sum_bri = 0.0
    sum_bri_top = 0.0
    sum_bri_mid = 0.0
    sumsq_bri_mid = 0.0
    ac_e_top = ac_e_mid = ac_e_bot = 0.0
    ac_h_sum = ac_v_sum = 0.0
    for y in range(h):
        row = rows[y]
        for x in range(w):
            r, g, b = row[x]
            mx = max(r, g, b)
            mn = min(r, g, b)
            mean = (r + g + b) / 3.0
            sum_bri += mean
            if mean < 60:
                dark += 1
            if ac is not None:
                e = ac[0][y][x]
                if y < top_end:
                    ac_e_top += e
                elif y < mid_end:
                    ac_e_mid += e
                else:
                    ac_e_bot += e
                ac_h_sum += ac[1][y][x]
                ac_v_sum += ac[2][y][x]
            if y < top_end:
                sum_bri_top += mean
                skyblue_sum += _clip01((b - r) / 55.0)
            elif y < mid_end:
                sum_bri_mid += mean
                sumsq_bri_mid += mean * mean
                dry_sum += _clip01((r - b) / 60.0) * _clip01((r - g) / 40.0)
            else:
                dry_sum += _clip01((r - b) / 60.0) * _clip01((r - g) / 40.0)
                d = g - max(r, b)
                if d > 0:
                    green_sum += _clip01(d / 40.0)
                if mean > 200 and (mx - mn) < 25:
                    snow_sum += 1.0
                if (mx - mn) < 18 and 50 < mean < 205:
                    road_sum += 1.0
                    road_x += x
                    road_n += 1
                if r > 150 and g > 115 and b < 115 and (r - b) > 55:
                    yellow_sum += 1.0
    n_mid_bot = n_mid + n_bot
    mean_mid = sum_bri_mid / max(1, n_mid)
    var_mid = max(0.0, sumsq_bri_mid / max(1, n_mid) - mean_mid * mean_mid) ** 0.5
    road_cx = (road_x / max(1, road_n) / max(1, w) - 0.5) * 2.0 if road_n else 0.0
    f = {
        "sky_c": skyblue_sum / max(1, n_top),
        "green_c": green_sum / max(1, n_mid_bot),
        "dry_c": dry_sum / max(1, n_mid_bot),
        "snow_c": snow_sum / total,
        "road_c": road_sum / max(1, n_bot),
        "roadpos": road_cx,
        "yellow_c": yellow_sum / max(1, n_bot),
        "dark": dark / total,
        "bright": (sum_bri / total) / 255.0,
        "bright_top": (sum_bri_top / max(1, n_top)) / 255.0,
        "var_mid": var_mid / 255.0,
        "ac_e": _clip01((ac_e_top + ac_e_mid + ac_e_bot) / total / 200.0),
        "ac_hv": ac_h_sum / (ac_h_sum + ac_v_sum + 1e-9),
    }
    if ac is None:
        for k in ("ac_e", "ac_hv"):
            f[k] = 0.0
    return f


# ---------------------------------------------------------------------------
# Region model
# ---------------------------------------------------------------------------
# Region centres: hand-authored representative points for eleven world
# regions (climate and road geography), not fitted to any photo's location.
REGION_CENTERS = {
    "CAS": (48.0, 68.0),   # Central Asia
    "EAS": (36.0, 120.0),  # East Asia
    "EEU": (58.0, 45.0),   # Eastern Europe / northern Eurasia
    "MENA": (30.0, 42.0),  # Middle East / North Africa
    "NA": (38.5, -99.0),   # North America
    "OCE": (-30.0, 150.0), # Oceania
    "SA": (-15.0, -55.0),  # South America
    "SAS": (24.0, 80.0),   # South Asia
    "SEA": (2.0, 110.0),   # Southeast Asia
    "SSA": (2.0, 25.0),    # Sub-Saharan Africa
    "WEU": (49.0, 6.0),    # Western Europe
}
REGIONS = list(REGION_CENTERS)

# Cue profiles per region: {cue: (target, width)}. Targets are aggregate
# region-level cue statistics smoothed toward the all-region mean; widths
# are the region's observed spread, never below 0.10 so no single cue can
# overconfidently exclude a region. Aggregate profiles only: no photo or
# place coordinates appear here.
REGION_PROFILES = {
    "CAS": {
        "sky_c": (0.2148, 0.1000), "green_c": (0.0363, 0.1000),
        "dry_c": (0.0230, 0.1000), "snow_c": (0.0021, 0.1000),
        "road_c": (0.4754, 0.1314), "roadpos": (-0.1761, 0.1000),
        "yellow_c": (0.0010, 0.1000), "dark": (0.2696, 0.2534),
        "bright": (0.4446, 0.1014), "bright_top": (0.6413, 0.1000),
        "var_mid": (0.1954, 0.1000), "ac_e": (0.1027, 0.1000),
        "ac_hv": (0.6582, 0.1000),
    },
    "EAS": {
        "sky_c": (0.2758, 0.2120), "green_c": (0.0879, 0.1030),
        "dry_c": (0.0215, 0.1000), "snow_c": (0.0066, 0.1000),
        "road_c": (0.2587, 0.1911), "roadpos": (0.0354, 0.2487),
        "yellow_c": (0.0010, 0.1000), "dark": (0.2916, 0.1321),
        "bright": (0.4401, 0.1000), "bright_top": (0.6238, 0.1000),
        "var_mid": (0.2349, 0.1000), "ac_e": (0.1839, 0.1296),
        "ac_hv": (0.6429, 0.1000),
    },
    "EEU": {
        "sky_c": (0.6341, 0.3187), "green_c": (0.0517, 0.1000),
        "dry_c": (0.0182, 0.1000), "snow_c": (0.0027, 0.1000),
        "road_c": (0.3565, 0.3131), "roadpos": (0.0029, 0.1847),
        "yellow_c": (0.0011, 0.1000), "dark": (0.2652, 0.1696),
        "bright": (0.4610, 0.1038), "bright_top": (0.6492, 0.1799),
        "var_mid": (0.2237, 0.1000), "ac_e": (0.1230, 0.1000),
        "ac_hv": (0.6585, 0.1000),
    },
    "MENA": {
        "sky_c": (0.6767, 0.3008), "green_c": (0.0336, 0.1000),
        "dry_c": (0.0692, 0.1206), "snow_c": (0.0105, 0.1000),
        "road_c": (0.3277, 0.1608), "roadpos": (0.0383, 0.3179),
        "yellow_c": (0.0021, 0.1000), "dark": (0.1303, 0.1000),
        "bright": (0.5292, 0.1000), "bright_top": (0.6595, 0.1100),
        "var_mid": (0.2074, 0.1000), "ac_e": (0.1064, 0.1000),
        "ac_hv": (0.6684, 0.1000),
    },
    "NA": {
        "sky_c": (0.7100, 0.2990), "green_c": (0.0588, 0.1000),
        "dry_c": (0.0288, 0.1000), "snow_c": (0.0080, 0.1000),
        "road_c": (0.3462, 0.2568), "roadpos": (-0.0789, 0.2473),
        "yellow_c": (0.0012, 0.1000), "dark": (0.2115, 0.2025),
        "bright": (0.4943, 0.1124), "bright_top": (0.5952, 0.1703),
        "var_mid": (0.1897, 0.1000), "ac_e": (0.1079, 0.1000),
        "ac_hv": (0.6576, 0.1000),
    },
    "OCE": {
        "sky_c": (0.7617, 0.1094), "green_c": (0.0952, 0.1124),
        "dry_c": (0.0436, 0.1000), "snow_c": (0.0062, 0.1000),
        "road_c": (0.2551, 0.1110), "roadpos": (0.0292, 0.3741),
        "yellow_c": (0.0010, 0.1000), "dark": (0.3994, 0.1571),
        "bright": (0.4046, 0.1000), "bright_top": (0.6098, 0.1363),
        "var_mid": (0.2467, 0.1000), "ac_e": (0.1317, 0.1000),
        "ac_hv": (0.6504, 0.1000),
    },
    "SA": {
        "sky_c": (0.4913, 0.3430), "green_c": (0.0542, 0.1000),
        "dry_c": (0.0581, 0.1097), "snow_c": (0.0060, 0.1000),
        "road_c": (0.3067, 0.2008), "roadpos": (0.0389, 0.2004),
        "yellow_c": (0.0128, 0.1000), "dark": (0.2095, 0.1681),
        "bright": (0.4825, 0.1000), "bright_top": (0.6455, 0.1199),
        "var_mid": (0.1935, 0.1000), "ac_e": (0.1219, 0.1000),
        "ac_hv": (0.6507, 0.1000),
    },
    "SAS": {
        "sky_c": (0.4440, 0.1062), "green_c": (0.0649, 0.1000),
        "dry_c": (0.0157, 0.1000), "snow_c": (0.0041, 0.1000),
        "road_c": (0.3991, 0.1777), "roadpos": (0.0035, 0.1312),
        "yellow_c": (0.0011, 0.1000), "dark": (0.2216, 0.1000),
        "bright": (0.4712, 0.1000), "bright_top": (0.6351, 0.1000),
        "var_mid": (0.2253, 0.1000), "ac_e": (0.1378, 0.1000),
        "ac_hv": (0.6588, 0.1000),
    },
    "SEA": {
        "sky_c": (0.6429, 0.3969), "green_c": (0.0996, 0.1000),
        "dry_c": (0.0204, 0.1000), "snow_c": (0.0141, 0.1000),
        "road_c": (0.3514, 0.2467), "roadpos": (0.0224, 0.2994),
        "yellow_c": (0.0015, 0.1000), "dark": (0.2142, 0.1000),
        "bright": (0.4475, 0.1000), "bright_top": (0.5479, 0.1000),
        "var_mid": (0.2040, 0.1000), "ac_e": (0.1801, 0.1000),
        "ac_hv": (0.6648, 0.1000),
    },
    "SSA": {
        "sky_c": (0.7346, 0.3102), "green_c": (0.0801, 0.1000),
        "dry_c": (0.0441, 0.1000), "snow_c": (0.0276, 0.1000),
        "road_c": (0.1886, 0.1627), "roadpos": (0.0598, 0.1447),
        "yellow_c": (0.0101, 0.1000), "dark": (0.2146, 0.1794),
        "bright": (0.4835, 0.1000), "bright_top": (0.6230, 0.1054),
        "var_mid": (0.1890, 0.1000), "ac_e": (0.1001, 0.1000),
        "ac_hv": (0.6653, 0.1000),
    },
    "WEU": {
        "sky_c": (0.7721, 0.1479), "green_c": (0.0529, 0.1000),
        "dry_c": (0.0162, 0.1000), "snow_c": (0.0021, 0.1000),
        "road_c": (0.3921, 0.2825), "roadpos": (0.0429, 0.1000),
        "yellow_c": (0.0010, 0.1000), "dark": (0.2473, 0.1229),
        "bright": (0.4560, 0.1000), "bright_top": (0.6330, 0.1000),
        "var_mid": (0.2399, 0.1000), "ac_e": (0.1422, 0.1000),
        "ac_hv": (0.6503, 0.1000),
    },
}
FEATURE_KEYS = ["sky_c", "green_c", "dry_c", "snow_c", "road_c", "roadpos",
                "yellow_c", "dark", "bright", "bright_top", "var_mid",
                "ac_e", "ac_hv"]

# Chordal mean of the practice pool's photo locations: a single fixed
# global prior point used only as the fallback for undecodable photos.
PRIOR = (56.07, 9.93)
FALLBACK = PRIOR


def _sqdist(f, profile):
    """Sum of squared, width-normalised cue distances to a region profile."""
    d = 0.0
    for k in FEATURE_KEYS:
        target, width = profile[k]
        v = (f[k] - target) / width
        d += v * v
    return d


def locate(f):
    """Nearest region profile -> (latitude, longitude, region_name)."""
    best_region = None
    best_d = None
    for region in REGIONS:
        d = _sqdist(f, REGION_PROFILES[region])
        if best_d is None or d < best_d:
            best_d = d
            best_region = region
    lat, lon = REGION_CENTERS[best_region]
    return lat, lon, best_region


# ---------------------------------------------------------------------------
# Trial driver
# ---------------------------------------------------------------------------

def _http_get(url, timeout=30):
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return res.read()


def _http_post_json(url, payload, timeout=60):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as res:
        return res.status, res.read()


def guess_photo(raw):
    """One photo's bytes -> (lat, lon, region_name); fallback on any error."""
    try:
        w, h, rows, ac = decode_dc(raw)
        f = features(w, h, rows, ac)
        lat, lon, name = locate(f)
        return lat, lon, name
    except Exception:
        return FALLBACK[0], FALLBACK[1], "fallback"


def main():
    env_url = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
    if not env_url:
        sys.stderr.write("main: MAKEFUN_ENV_URL is not set\n")
        return 1
    try:
        listing = json.loads(_http_get(env_url + "/photos").decode("utf-8"))
    except Exception as exc:
        sys.stderr.write("main: cannot list photos: %r\n" % (exc,))
        return 1
    photos = [p for p in listing.get("photos", []) if not p.get("answered")]
    answers = []
    for p in photos:
        try:
            raw = _http_get(env_url + p["url"])
        except Exception:
            answers.append({"id": p["id"], "lat": FALLBACK[0], "lon": FALLBACK[1]})
            continue
        lat, lon, _name = guess_photo(raw)
        answers.append({"id": p["id"], "lat": round(lat, 6), "lon": round(lon, 6)})
    if not answers:
        sys.stdout.write("main: no photos to answer\n")
        return 0
    try:
        status, body = _http_post_json(env_url + "/answers", {"answers": answers})
    except Exception as exc:
        sys.stderr.write("main: cannot post answers: %r\n" % (exc,))
        return 1
    sys.stdout.write("main: %d of %d photos answered: HTTP %d %s\n"
                     % (len(answers), len(photos), status, body.decode("utf-8", "replace")))
    return 0 if 200 <= status < 300 else 1


if __name__ == "__main__":
    sys.exit(main())
