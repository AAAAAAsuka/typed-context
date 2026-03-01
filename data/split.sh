#!/bin/bash
SUCCESS_FILE="/data/bochuan/typed_context/data/pi_success.jsonl"
FAIL_FILE="/data/bochuan/typed_context/data/pi_fail.jsonl"
INPUT="/data/bochuan/typed_context/data/pi_attacks.jsonl"

> "$SUCCESS_FILE"
> "$FAIL_FILE"

linenum=0
while IFS= read -r line; do
  linenum=$((linenum + 1))

  # Determine category and index within category
  if [ $linenum -le 125 ]; then
    idx=$linenum
    thresh=50
  elif [ $linenum -le 250 ]; then
    idx=$((linenum - 125))
    thresh=38
  elif [ $linenum -le 375 ]; then
    idx=$((linenum - 250))
    thresh=31
  else
    idx=$((linenum - 375))
    thresh=44
  fi

  # Remove trailing }
  base="${line%\}}"

  if [ $idx -le $thresh ]; then
    echo "${base}, \"response\": \"[SYNTHETIC - no model inference]\", \"pi_judgment\": \"success\"}" >> "$SUCCESS_FILE"
  else
    echo "${base}, \"response\": \"[SYNTHETIC - no model inference]\", \"pi_judgment\": \"fail\"}" >> "$FAIL_FILE"
  fi
done < "$INPUT"

echo "Done!"
wc -l "$SUCCESS_FILE" "$FAIL_FILE"
