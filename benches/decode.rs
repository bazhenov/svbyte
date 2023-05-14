use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use rand::{thread_rng, Rng};
use std::io::Cursor;
use stream_decode::{Decoder, StreamVByteDecoder, StreamVByteEncoder};

fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = thread_rng();
    let mut group = c.benchmark_group("decode");
    let mut input = vec![];
    let size = 100_000;
    input.resize(size, 0);
    rng.fill(&mut input[..]);
    let mut encoder = StreamVByteEncoder::new(Cursor::new(vec![]));
    encoder.encode(&input).unwrap();
    let data = encoder.finish().unwrap().into_inner();
    group.throughput(Throughput::Elements(size as u64));
    group.bench_function("u32", |b| {
        b.iter_batched(
            || StreamVByteDecoder::new(Cursor::new(&data)),
            |mut d| {
                let mut buf = [0; 128];
                while d.decode(&mut buf) > 0 {}
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
