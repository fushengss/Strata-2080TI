#!/usr/bin/env python3
"""one-off: fix the chunk<N>m<MIN> regular expression in /tmp/claude-parktest/setcfg.py (it had double backslashes)"""
p = "/tmp/claude-parktest/setcfg.py"
s = open(p).read()
bad = 'mm = re.match(r"chunk(\\\\d+)(?:m(\\\\d+))?", v)'
good = 'mm = re.match(r"chunk(\\d+)(?:m(\\d+))?", v)'
print("bad form present:", bad in s)
s = s.replace(bad, good)
open(p, "w").write(s)
import re
print([line.strip() for line in s.splitlines() if "re.match" in line])
print(re.match(r"chunk(\d+)(?:m(\d+))?", "chunk4m256").groups(), re.match(r"chunk(\d+)(?:m(\d+))?", "chunk2").groups())
