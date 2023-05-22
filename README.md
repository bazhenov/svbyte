## Rust Stream VByte encoder/decoder

This library provides encoding/decoding primitives for Stream VByte encoding.

Stream VByte encoding is a SIMD accelerated algorithm of VarInt decompression. It is used in a search and database
systems as a way of efficiently store and stream large number of VarInts from a disk or main memory.

Benchmarking results:

| CPU             | Base Freq. (GHz)  | Turbo Freq. (GHz) | Result (GElem/s) |
|-----------------|------------------:|------------------:|-----------------:|
| Xeon E3-1245 v5 |               3.5 | 3.9               | 5.0              |
| Core i7-1068NG7 |               2.3 | 4.1               | 5.5              |

The idea behind VarInt is not to store leading zero bytes of the number. This way large amount of relatively small
numbers can be stored in a much more compact way. VarInt encoding is frequently used with delta-encoding if numbers
are stored in the ascending order. This way all the numbers are smaller by magnitude, hence better compression.

Stream VByte working using two data streams: control stream and data stream. Control stream contains control words (1
byte each). Each control word describe length of 4 numbers in the data stream (2 bits per number, `00` - length 1,
`01` - length 2 and so on).

- [API Documentation](https://docs.rs/svbyte/latest/svbyte/)
- [Examples](https://github.com/bazhenov/svbyte/tree/master/examples)

## Links

- [Stream VByte: Faster Byte-Oriented Integer Compression][pub] by Daniel Lemire and Leonid Boytsov.
- [Stream VByte: breaking new speed records for integer compression][blog-post] by Daniel Lemire
- [Compress-a-Palooza: Unpacking 5 Billion Varints in only 4 Billion CPU Cycles][bazhenov-blog-post]

[pub]: https://arxiv.org/abs/1709.08990
[blog-post]: https://lemire.me/blog/2017/09/27/stream-vbyte-breaking-new-speed-records-for-integer-compression/
[bazhenov-blog-post]: https://www.bazhenov.me/posts/rust-stream-vbyte-varint-decoding/
