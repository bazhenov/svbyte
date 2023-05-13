use std::{io::Cursor, mem::size_of};

use criterion::{criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use rand::{thread_rng, Rng};
use stream_decode::StreamVByteEncoder;

fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = thread_rng();
    let mut group = c.benchmark_group("encode");
    let mut input = vec![];
    let size = 1000;
    input.resize(size, 0);
    rng.fill(&mut input[..]);
    group.throughput(Throughput::Bytes((size * size_of::<u32>()) as u64));
    group.bench_function("u32", |b| {
        b.iter_batched(
            || StreamVByteEncoder::new(Cursor::new(vec![])),
            |mut e| {
                e.encode(&input);
                e.finish()
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
