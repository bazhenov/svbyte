use criterion::{criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use rand::{thread_rng, Rng, RngCore};
use std::io::Cursor;
use stream_decode::{Decoder, MemorySegments, StreamVByteDecoder, StreamVByteEncoder};

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode");
    let size = 100_000;
    let input = generate_random_data(size);
    let mut encoder = StreamVByteEncoder::new(Cursor::new(vec![]));
    encoder.encode(&input).unwrap();
    let data = encoder.finish().unwrap().into_inner();
    group.throughput(Throughput::Elements(size as u64));
    group.bench_function("u32", |b| {
        b.iter_batched(
            || StreamVByteDecoder::new(MemorySegments::new(&data)).unwrap(),
            |mut d| {
                let mut buf = [0; 256];
                while d.decode(&mut buf) > 0 {}
            },
            BatchSize::SmallInput,
        )
    });
    group.finish();
}

fn generate_random_data(size: usize) -> Vec<u32> {
    let mut rng = thread_rng();
    let mut input = vec![];
    input.resize(size, 0);
    for i in input.iter_mut() {
        *i = match rng.gen_range(1..=4) {
            1 => rng.next_u32() % 0xFF,
            2 => rng.next_u32() % 0xFFFF,
            3 => rng.next_u32() % 0xFFFFFF,
            _ => rng.next_u32(),
        }
    }
    input
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
