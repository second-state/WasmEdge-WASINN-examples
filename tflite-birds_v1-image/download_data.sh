FIXTURE=https://raw.githubusercontent.com/second-state/wasm-learning/master/rust/birds_v1/
TODIR=$1

if [ ! -f $TODIR/bird.jpg ]; then
    wget --no-clobber --directory-prefix=$TODIR $FIXTURE/bird.jpg
fi
