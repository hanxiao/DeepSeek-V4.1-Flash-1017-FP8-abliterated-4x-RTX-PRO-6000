#!/bin/bash
# VM side, root. Rebuild codebench contexts (sglang source files) into /var/lib/ds41/cb and /tmp/ds41cb.
# Original ctx_*.txt (122,595 / 246,710 / 493,991 bytes) were lost when a reboot cleared /tmp.
set -e
W=/var/lib/ds41/cb; mkdir -p $W; cd $W
[ -d sglang ] || git clone -q --depth 1 https://github.com/sgl-project/sglang.git sglang
S=sglang/python/sglang/srt
git -C sglang rev-parse HEAD > $W/sglang.rev
FIRST="$S/mem_cache/radix_cache.py $S/function_call/qwen3_coder_detector.py"
REST=$( (ls $S/mem_cache/*.py $S/function_call/*.py $S/managers/*.py $S/sampling/*.py 2>/dev/null) | sort | grep -v -e radix_cache.py -e qwen3_coder_detector.py)
for tgt in 30:122595 60:246710 120:493991; do
  n=${tgt%%:*}; lim=${tgt##*:}; out=$W/ctx_$n.txt; : > $out
  for f in $FIRST $REST; do
    p=${f#sglang/}
    sz=$(( $(stat -c %s $f) + ${#p} + 30 ))
    cur=$(stat -c %s $out)
    [ $((cur + sz)) -gt $lim ] && continue
    printf '<file path="%s">\n' "$p" >> $out; cat $f >> $out; printf '</file>\n' >> $out
  done
  echo "ctx_$n $(stat -c %s $out) bytes, $(grep -c '^<file path' $out) files"
done
mkdir -p /tmp/ds41cb; cp $W/ctx_*.txt /tmp/ds41cb/; chmod 644 /tmp/ds41cb/* $W/ctx_*.txt
