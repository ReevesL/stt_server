# stt_server

Started: April 5, 2026

---

## History
**October 16, 2026**
* 1.5.0: switched to large-v3-turbo and passed thread count to faster-whisper (was stuck at 4)
* traced a vanished job to a CTranslate2 segfault at 24 threads; benchmarked 4/12/18 threads
* 1.5.1: default 14 threads and 1 worker; strip accents from output filenames to fix SMB access on Mac
