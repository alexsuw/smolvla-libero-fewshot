#!/usr/bin/env bash
# Build the pinned FFmpeg as shared libraries so TorchCodec can load it.
set -euo pipefail

VERSION=7.1.1
ARCHIVE_SHA256=733984395e0dbbe5c046abda2dc49a5544e7e0e1e2366bba849222ae9e3a03b1

usage() {
  cat <<'EOF'
Usage: scripts/build_shared_ffmpeg.sh

Build checksummed FFmpeg 7.1.1 shared libraries under VLA_CACHE_DIR so
TorchCodec can decode training videos. Existing verified output is reused.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -ne 0 ]]; then
  usage >&2
  exit 2
fi

: "${VLA_CACHE_DIR:?VLA_CACHE_DIR is required}"

root="${VLA_FFMPEG_SHARED_ROOT:-${VLA_CACHE_DIR}/ffmpeg-shared-${VERSION}}"
source_root="${root}/src"
prefix="${root}/install"
archive="${source_root}/ffmpeg-${VERSION}.tar.xz"
source_dir="${source_root}/ffmpeg-${VERSION}"
jobs="${VLA_BUILD_JOBS:-$(getconf _NPROCESSORS_ONLN)}"

for command in curl tar make gcc pkg-config sha256sum; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "$command is required; shared FFmpeg was not built." >&2
    exit 2
  fi
done
if ! pkg-config --exists aom; then
  echo "libaom development files are required; shared FFmpeg was not built." >&2
  exit 2
fi

mkdir -p "$source_root" "$prefix"
if [[ -f "${prefix}/lib/libavutil.so.59" ]]; then
  LD_LIBRARY_PATH="${prefix}/lib:${LD_LIBRARY_PATH:-}" \
    "${prefix}/bin/ffmpeg" -version
  printf 'export PATH="%s/bin:$PATH"\n' "$prefix"
  printf 'export LD_LIBRARY_PATH="%s/lib:${LD_LIBRARY_PATH:-}"\n' "$prefix"
  exit 0
fi

if [[ ! -f "$archive" ]]; then
  curl -fL --retry 3 \
    "https://ffmpeg.org/releases/ffmpeg-${VERSION}.tar.xz" \
    -o "$archive"
fi
printf '%s  %s\n' "$ARCHIVE_SHA256" "$archive" | sha256sum -c -

if [[ ! -d "$source_dir" ]]; then
  tar -xf "$archive" -C "$source_root"
fi

(
  cd "$source_dir"
  ./configure \
    --prefix="$prefix" \
    --enable-shared \
    --disable-static \
    --enable-pic \
    --enable-gpl \
    --enable-libaom \
    --disable-doc \
    --disable-ffplay \
    --disable-debug \
    --extra-ldflags="-Wl,-rpath,${prefix}/lib"
  make -j"$jobs"
  make install
)

test -f "${prefix}/lib/libavutil.so.59"
test -f "${prefix}/lib/libavcodec.so.61"
printf 'export PATH="%s/bin:$PATH"\n' "$prefix"
printf 'export LD_LIBRARY_PATH="%s/lib:${LD_LIBRARY_PATH:-}"\n' "$prefix"
