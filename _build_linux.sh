#!/bin/bash
set -e

# === User-configurable options ===
cc=120
# deprecatedGpu=true
# define=mul_Mod_ptx

# === Environment setup ===
cd "$(dirname "$0")"

# === Derived build options ===
NVCC=nvcc
GENCODE="-gencode=arch=compute_${cc},code=\"sm_${cc},compute_${cc}\""
DEFINE_OPT=""
if [[ -n "$define" ]]; then
    DEFINE_OPT="-D$define"
fi
DEPRECATED_OPT=""
if [[ "$deprecatedGpu" == "true" ]]; then
    DEPRECATED_OPT="--Wno-deprecated-gpu-targets"
fi

OPTIONS=($GENCODE $DEFINE_OPT $DEPRECATED_OPT --use-local-env --keep-dir linux --machine 64 --compile -cudart static -Xcompiler -O2)
OUTDIR=linux/obj

# === Create output directory if it doesn't exist ===
mkdir -p "$OUTDIR"

# === Initialize object file list ===
LINK_OBJS=()

echo "=== Compiling all .cu and .cpp files ==="

while IFS= read -r -d '' SRC; do
    REL=${SRC#"$PWD"/}
    OBJ="$OUTDIR/${REL}.o"
    OBJ_DIR=$(dirname "$OBJ")
    mkdir -p "$OBJ_DIR"

    echo "Compiling: $SRC → $OBJ"
    $NVCC "${OPTIONS[@]}" "$SRC" -o "$OBJ"
    LINK_OBJS+=("$OBJ")
done < <(find . -type f \( -name "*.cu" -o -name "*.cpp" \) -print0)

echo
echo "=== Linking into linux/LucasNTT ==="
$NVCC $DEPRECATED_OPT -link -o linux/LucasNTT $GENCODE --machine 64 "${LINK_OBJS[@]}"

echo
echo "=== Build successful ==="
