#!/usr/bin/env bash
# Phase 7 tool tests with synthetic data through replay (T4, T6) or from files (T5), and T5 on the real scene.
#   phase7_tools.sh [t4] [t5] [t6] [t5real]
source "$(dirname "$0")/common.sh"
TESTS=${*:-t4 t5 t6 t5real}
STL=$TRK_DIR/../models/test_object.stl
POSE=0.0,0.0,0.42,0.92,0.2,0.25,0.22

t_t4() {
    for cov in A B; do
        local d=$LOGROOT/t4_cover_$cov; rm -rf "$d"; mon_start "$d"
        local F=$DATA/t4_cover_$cov.trk
        $B/trk_synth --out $F --frames 600 --pose $POSE --noise-mm 1 --blank $cov:250:600 --stl $STL > "$d/synth.log" 2>&1
        cp $TCONF "$d/config.json"
        # the configuration starts with the serials swapped, so a correct result must change it
        python3 - "$d/config.json" <<'PY'
import sys, re
p = sys.argv[1]; t = open(p).read()
t = t.replace('"serial": "234322070408"', '"serial": "X1"').replace('"serial": "234322070144"', '"serial": "234322070408"').replace('"serial": "X1"', '"serial": "234322070144"')
open(p, "w").write(t)
PY
        $B/trk_replay --in $F --start-delay 1.5 > "$d/replay.log" 2>&1 &
        local rp=$!
        $B/trk_camid --config "$d/config.json" --source shm:/d435_pairs --timed 1.5 --write > "$d/camid.log" 2>&1
        echo "camid_rc=$?" >> "$d/meta.txt"
        kill $rp 2>/dev/null; wait $rp 2>/dev/null
        mon_stop "$d"
        local s1; s1=$(python3 -c "import json;print(json.load(open('$d/config.json'))['cameras']['D435i_1']['serial'])")
        local expect=234322070408; [ $cov = B ] && expect=234322070144
        echo "t4 cover $cov: D435i_1 written = $s1, expected $expect -> $([ "$s1" = "$expect" ] && echo PASS || echo FAIL)" | tee -a "$d/result.txt"
    done
}

t_t5() {
    # (a) camera B's true transform differs from the nominal one by 3 deg and 8 mm; T5 with the nominal configuration
    local d=$LOGROOT/t5_synth_correction; rm -rf "$d"; mkdir -p "$d"
    python3 - "$d/true_calib.json" "$d/flip_calib.json" <<'PY'
import json, math, sys
def mat(Rr, t): return [[Rr[0][0],Rr[0][1],Rr[0][2],t[0]],[Rr[1][0],Rr[1][1],Rr[1][2],t[1]],[Rr[2][0],Rr[2][1],Rr[2][2],t[2]],[0,0,0,1]]
def mul(A,B): return [[sum(A[i][k]*B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
def rot(ax, ang):
    n = math.sqrt(sum(a*a for a in ax)); x,y,z = (a/n for a in ax); c, s = math.cos(ang), math.sin(ang); C = 1-c
    return [[c+x*x*C, x*y*C-z*s, x*z*C+y*s],[y*x*C+z*s, c+y*y*C, y*z*C-x*s],[z*x*C-y*s, z*y*C+x*s, c+z*z*C]]
R1 = [[-1,0,0],[0,-0.906307787,0.422618262],[0,0.422618262,0.906307787]]; t1 = [0.0175,-0.240664,0.0561495]
R2 = [[1,0,0],[0,0.906307787,-0.422618262],[0,0.422618262,0.906307787]]; t2 = [-0.0175,0.240664,0.0561495]
R2p = mul(rot([1,2,-1], math.radians(3.0)), R2); t2p = [t2[0]+0.005, t2[1]-0.004, t2[2]+0.0047]
json.dump({"version":1,"cameras":{"D435i_1":{"T_rig_from_camera":mat(R1,t1)},"D435i_2":{"T_rig_from_camera":mat(R2p,t2p)}}}, open(sys.argv[1],"w"), indent=1)
F = [[-1,0,0],[0,-1,0],[0,0,1]]   # true camera frames rotated 180 deg about z relative to the assumption
json.dump({"version":1,"cameras":{"D435i_1":{"T_rig_from_camera":mat(mul(R1,F),t1)},"D435i_2":{"T_rig_from_camera":mat(mul(R2,F),t2)}}}, open(sys.argv[2],"w"), indent=1)
PY
    $B/trk_synth --out $DATA/t5_true.trk --frames 300 --pose $POSE --noise-mm 1 --calib "$d/true_calib.json" --stl $STL > "$d/synth.log" 2>&1
    $B/trk_calib --out "$d/calib.json" --config $TCONF --source $DATA/t5_true.trk --pairs 300 > "$d/calib.log" 2>&1
    python3 - "$d/calib.json" "$d/true_calib.json" <<'PY' | tee "$d/result.txt"
import json, math, sys
c = json.load(open(sys.argv[1]))["cameras"]["D435i_2"]["T_rig_from_camera"]; t = json.load(open(sys.argv[2]))["cameras"]["D435i_2"]["T_rig_from_camera"]
dp = math.sqrt(sum((c[i][3]-t[i][3])**2 for i in range(3)))*1000
tr = sum(c[k][i]*t[k][i] for i in range(3) for k in range(3)); ang = math.degrees(math.acos(max(-1,min(1,(tr-1)/2))))
print(f"t5 correction: corrected B vs true B: {dp:.2f} mm, {ang:.3f} deg -> {'PASS' if dp <= 2.0 and ang <= 0.5 else 'FAIL'} (limits 2 mm, 0.5 deg)")
PY
    # (b) camera frames rotated 180 deg about z: T5 must find that assumption as the cause
    local e=$LOGROOT/t5_synth_axis; rm -rf "$e"; mkdir -p "$e"; cp "$d/flip_calib.json" "$e/"
    $B/trk_synth --out $DATA/t5_flip.trk --frames 300 --pose $POSE --noise-mm 1 --calib "$e/flip_calib.json" --stl $STL > "$e/synth.log" 2>&1
    $B/trk_calib --out "$e/calib.json" --config $TCONF --source $DATA/t5_flip.trk --pairs 300 > "$e/calib.log" 2>&1
    if grep -q "closest assumption: camera frame rotated 180 deg about z" "$e/calib.log"; then r=PASS; else r=FAIL; fi
    echo "t5 axis convention: $(grep -E 'RESULT' "$e/calib.log") -> $r" | tee "$e/result.txt"
}

t_t6() {
    local d=$LOGROOT/t6_synth; rm -rf "$d"; mon_start "$d"
    # empty scene: background plane at z = 0.55 m (behind the object as the upward-looking cameras see it), no object;
    # then the object in front of the plane
    $B/trk_synth --out $DATA/t6_empty.trk --frames 120 --pose 0,0,0.40,1,0,0,0 --absent 0:120 --plane-z 0.55 --noise-mm 1 --stl $STL > "$d/synth_empty.log" 2>&1
    $B/trk_synth --out $DATA/t6_object.trk --frames 200 --pose 0.01,0.0,0.40,0.92,0.2,0.25,0.22 --plane-z 0.55 --noise-mm 1 --seed 3 --stl $STL > "$d/synth_object.log" 2>&1
    $B/trk_replay --in $DATA/t6_empty.trk --start-delay 1.5 > "$d/replay.log" 2>&1 &
    local rp=$!
    $B/trk_bgmap --out "$d/bgmap.bin" --config $TCONF --source shm:/d435_pairs --pairs 100 > "$d/bgmap.log" 2>&1
    echo "bgmap_rc=$?" >> "$d/meta.txt"
    kill $rp 2>/dev/null; wait $rp 2>/dev/null
    python3 - $TCONF "$d/config_map.json" "$d/bgmap.bin" <<'PY'
import json, sys
c = json.load(open(sys.argv[1])); c["background_map"] = sys.argv[3]
c["model_file"] = sys.argv[1].rsplit("/", 2)[0] + "/../models/test_object.model"; c["mesh_file"] = sys.argv[1].rsplit("/", 2)[0] + "/../models/test_object.stl"
json.dump(c, open(sys.argv[2], "w"), indent=1)
PY
    $B/trk_offline --in $DATA/t6_object.trk --out "$d/without_map.csv" --config $TCONF > "$d/offline_without.log" 2>&1
    $B/trk_offline --in $DATA/t6_object.trk --out "$d/with_map.csv" --config "$d/config_map.json" > "$d/offline_with.log" 2>&1
    $B/trk_clouddump --in $DATA/t6_object.trk --out "$d/cloud_object.csv" --index 10 --stride 2 --config $TCONF > /dev/null
    mon_stop "$d"
    python3 "$(dirname "$0")/check_bgmap.py" "$d/bgmap.bin" "$d/cloud_object.csv" "$d/without_map.csv" "$d/with_map.csv" | tee "$d/result.txt"
}

t_t5real() {
    local d=$LOGROOT/t5_real; rm -rf "$d"; mon_start "$d"
    $CAP --config $CONF --duration 25 --hist-out /dev/null --delay-hist-out /dev/null > "$d/capture.log" 2>&1 &
    local cp=$!
    sleep 7
    $B/trk_record --out $DATA/t5_real.trk --pairs 300 > "$d/record.log" 2>&1
    $B/trk_calib --out "$d/calib.json" --config $TCONF --source $DATA/t5_real.trk --pairs 300 > "$d/calib.log" 2>&1
    wait $cp
    mon_stop "$d"
    cat "$d/calib.log"
}

for t in $TESTS; do echo "=== $(date +%T) $t"; "t_$t"; done
echo "=== $(date +%T) PHASE7_DONE"
