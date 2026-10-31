#!/bin/bash
# Download GDP hourly (ERDDAP drifter_hourly_qc, v2.01) month by month for a lat/lon box.
# Usage: download_gdp_hourly.sh TAG LAT_MIN LAT_MAX LON_MIN LON_MAX FIRST_YYYY-MM LAST_YYYY-MM OUTDIR [PARALLEL]
# Files: OUTDIR/gdp1h_TAG_YYYY-MM.nc ; existing files are skipped, so a rerun only fetches missing months.
set -u
TAG=$1; LA0=$2; LA1=$3; LO0=$4; LO1=$5; FIRST=$6; LAST=$7; OUT=$8; PAR=${9:-4}
mkdir -p "$OUT"
V="ID,time,latitude,longitude,ve,vn,err_lat,err_lon,err_ve,err_vn,gap,location_type,drogue_lost_date"

one_month() {
  ym=$1
  out="$OUT/gdp1h_${TAG}_${ym}.nc"
  [ -s "$out" ] && { echo "$ym skip"; return 0; }
  y=${ym%-*}; m=$((10#${ym#*-}))
  if [ $m -eq 12 ]; then end=$(printf "%04d-01" $((y+1))); else end=$(printf "%04d-%02d" $y $((m+1))); fi
  Q="$V&time>=${ym}-01T00:00:00Z&time<${end}-01T00:00:00Z&latitude>=$LA0&latitude<=$LA1&longitude>=$LO0&longitude<=$LO1"
  Q=$(printf '%s' "$Q" | sed 's/>/%3E/g; s/</%3C/g')
  for t in 1 2 3 4; do
    code=$(curl -s --max-time 1200 -o "$out.part" -w "%{http_code}" \
      "https://erddap.aoml.noaa.gov/gdp/erddap/tabledap/drifter_hourly_qc.nc?$Q")
    if [ "$code" = "200" ]; then mv "$out.part" "$out"; echo "$ym ok $(stat -c %s "$out")"; return 0; fi
    echo "$ym try$t http=$code"; sleep $((10 * t))
  done
  rm -f "$out.part"; echo "$ym FAILED"
}
export -f one_month; export OUT TAG LA0 LA1 LO0 LO1 V

months() {
  y=${FIRST%-*}; m=$((10#${FIRST#*-}))
  while :; do
    ym=$(printf "%04d-%02d" $y $m); echo "$ym"
    [ "$ym" = "$LAST" ] && break
    m=$((m+1)); [ $m -gt 12 ] && { m=1; y=$((y+1)); }
  done
}
months | xargs -P "$PAR" -I{} bash -c 'one_month {}'
