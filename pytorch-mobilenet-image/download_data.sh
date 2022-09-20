FIXTURE=https://github.com/intel/openvino-rs/raw/v0.3.3/crates/openvino/tests/fixtures/mobilenet
TODIR=$1

if [ ! -f $TODIR/input.jpg ]; then
    wget --no-clobber --directory-prefix=$TODIR https://github.com/bytecodealliance/wasi-nn/raw/main/rust/examples/images/1.jpg -O input.jpg
fi
