# plutosdr-fft-log

![scanner.py](https://i.imgur.com/qIsP9PW.png)


*plutosdr-fft-log* is a tool for logging and playing back arbitrarily large portions of the RF spectrum. it can also identify peaks based on a handful of input parameters and log them to console

> **plutosdr-fft-log** uses the gnuradio python library, please refer to the [GNU Radio and IIO Device](https://wiki.analog.com/resources/tools-software/linux-software/gnuradio) page for more information

## `scanner.py`

`scanner.py` is designed to scan frequency ranges and log them efficiently. it is configurable via several command-line arguments:

- `--dir`: directory to save fft files (default: `./fft`).
- `--bandwidth`: bandwidth per fft in mhz. must be converted to hz within the application (default: 10 mhz).
- `--start`: start frequency in mhz, converted to hz within the application (default: 90 mhz).
- `--end`: end frequency in mhz, converted to hz within the application (default: 150 mhz).
- `--fftsize`: fft size, must be a power of 2 (default: 1024).
- `--frames`: number of fft frames to capture each sample period (default: 2).
- `--cutoff`: cutoff frequency for peaks. if false, defaults to 2 standard deviations above the mean (default: `false`).
- `--width`: minimum width for peak detection (default: 5).
- `--distance`: minimum distance between peaks (default: 30).

> there is no `--samplerate` argument as the plutosdr uses quadrature sampling -- it is capable of fulfilling the minimum nyquist sampling rate for its entire RF bandwidth


`scanner.py` continuously scans a specified frequency range and logs the FFT results to `.bin` files. each log file contains data for an arbitrary large portion of the spectrum, allowing continuous spectrum monitoring:

- FFT computations are performed for chunks of the spectrum defined by the `start`, `end`, `bandwidth`, `fftsize`, and `frames`
- the resulting fft data is logged to binary files in the specified directory. each file's name contains necessary metadata for identifying the scan parameters

example filename: `80_280_50_4096_8_1715568957.bin`, which encodes the start and end frequencies, the bandwidth, fft size, number of frames, and a timestamp.

## `replay.pl`

`replay.pl` plays back fft log files generated by `scanner.py`, printing detected peaks to the console:

- `--dir`: directory containing FFT log files
- `--cutoff`: cutoff frequency for peaks, defaults to 2 standard deviations above the mean if not specified
- `--width`: minimum width for peak detection (default: 5)
- `--distance`: minimum distance between peaks (default: 30)
- `--sleep`: sleep time in seconds between processing each file to simulate real-time playback (default: 1.0)

`replay.pl` reads fft log files from the specified directory, using the metadata encoded in each file's name to properly interpret the data:

- it extracts parameters like start and end frequencies, bandwidth, fft size, and number of frames directly from the filename
- the tool applies peak detection algorithms to identify significant peaks within each FFT data set, based on the user-defined parameters for cutoff frequency, width, and distance
- detected peaks are printed to the console, allowing users to analyze frequency spikes over time

## peak logging
the information about the peaks present is printed to console in both scripts:

```
Peak at 88.09 MHz, height -21.09
Peak at 90.91 MHz, height -33.05
Peak at 93.09 MHz, height -30.46
Peak at 93.76 MHz, height -25.09
Peak at 94.53 MHz, height -18.05
Peak at 95.30 MHz, height -35.39
Peak at 96.10 MHz, height -27.13
Peak at 96.92 MHz, height -28.99
```

these are calculated based on the FFT .bin files, so running `replay.py` with different parameters will result in different peaks being displayed
