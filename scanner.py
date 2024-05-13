#!/usr/bin/env python3.10


import os
import sys
import signal
import time
import numpy as np
from argparse import ArgumentParser
from gnuradio import blocks, gr, iio
from gnuradio.fft import logpwrfft, window
from scipy.signal import find_peaks, ricker, firwin, lfilter
import matplotlib.pyplot as plt
from gnuradio.fft import window
import traceback

pause = False
button = None

def signal_handler(sig, frame):
    sys.exit()

class scan_python(gr.top_block):
    def __init__(self, samp_rate, rf_bandwidth, freq, fft_size, output_file):
        gr.top_block.__init__(self, "Not titled yet", catch_exceptions=True)
        self.samp_rate = samp_rate
        self.rf_bandwidth = rf_bandwidth
        self.freq = freq
        self.fft_size = fft_size
        self.output_file = output_file

        self.logpwrfft_x_0 = logpwrfft.logpwrfft_c(
            sample_rate=samp_rate,
            fft_size=fft_size,
            ref_scale=1,
            frame_rate=420,
            avg_alpha=1.0,
            average=True,
            win=window.hann,
            shift=True
        )

        self.iio_pluto_source_0 = iio.fmcomms2_source_fc32('ip:192.168.2.1', [True, True], 32768)
        self.iio_pluto_source_0.set_frequency(freq)
        self.iio_pluto_source_0.set_samplerate(samp_rate)
        self.iio_pluto_source_0.set_gain_mode(0, 'fast_attack')
        self.iio_pluto_source_0.set_gain(0, 64)
        self.iio_pluto_source_0.set_quadrature(True)
        self.iio_pluto_source_0.set_rfdc(True)
        self.iio_pluto_source_0.set_bbdc(True)
        self.iio_pluto_source_0.set_filter_params('Off', '', 0, 0)

        self.blocks_file_sink_0 = blocks.file_sink(gr.sizeof_float*fft_size, output_file, True)
        self.blocks_file_sink_0.set_unbuffered(True)

        self.connect((self.iio_pluto_source_0, 0), (self.logpwrfft_x_0, 0))
        self.connect((self.logpwrfft_x_0, 0), (self.blocks_file_sink_0, 0))

    def start_and_wait_for_data(self, max_file_size):
        self.start()  

        while True:
            time.sleep(0.1)  
            if os.path.exists(self.output_file):  
                current_size = os.path.getsize(self.output_file)  
                if current_size >= max_file_size:  
                    if current_size > max_file_size:
                        self.truncate_file(max_file_size)  
                    break

        self.stop()  
        self.wait()  

    def truncate_file(self, max_file_size):
        try:
            with open(self.output_file, 'r+b') as f:
                f.seek(max_file_size)
                f.truncate()
                os.fsync(f.fileno())
        except Exception as e:
            print("Failed to truncate file:", str(e))

def scan_frequency_range(start_freq, end_freq, bandwidth, samplerate, fftsize, frames, output_file):
    num_steps = int((end_freq - start_freq) / bandwidth) + 1
    all_data = []

    for step in range(num_steps):
        current_freq = start_freq + step * bandwidth
        print(f"Scanning {int(current_freq / 1e6)} MHz")
        
        tb = scan_python(samplerate, bandwidth, current_freq, fftsize, output_file)
        tb.start_and_wait_for_data(fftsize * 4 * frames * (step + 1))
    
    tb.stop()
    tb.wait()

    with open(output_file, "rb") as file:
        all_data = np.fromfile(file, dtype=np.float32)
    
    all_fft_results = []
    for step in range(num_steps):
        start_index = fftsize * frames * step
        end_index = start_index + fftsize * frames
        if end_index > len(all_data):
            print("Error: Not enough data for step", step)
            break
        
        fft_mat = all_data[start_index:end_index].reshape(-1, fftsize)
        mean_fft = np.mean(fft_mat, axis=0)
        all_fft_results.extend(mean_fft)

    return all_fft_results

def adjust_bandwidth(start_freq, end_freq, init_bandwidth, max_bandwidth):
    total_range = end_freq - start_freq
    num_bands = total_range / init_bandwidth

    if num_bands.is_integer():
        return init_bandwidth
    
    print("Bandwidth does not divide evenly into the total range, adjusting...")

    if num_bands < 1:
        return total_range
    else:
        num_bands = round(num_bands)
    
    adjusted_bandwidth = total_range / num_bands
    
    if max_bandwidth is not None:
        adjusted_bandwidth = min(adjusted_bandwidth, max_bandwidth)

    if total_range % adjusted_bandwidth != 0:
        num_bands = int(total_range // adjusted_bandwidth) + 1
        adjusted_bandwidth = total_range / num_bands
    
    print("Adjusted bandwidth:", adjusted_bandwidth, "Hz")

    return int(adjusted_bandwidth)

def update_data(args, line, scatter, ax, fig):
    if not pause:
        ts = int(time.time())
        
        if not os.path.exists(args.dir):
            os.makedirs(args.dir)

        output_file = os.path.join(args.dir, f'{int(args.start/1e6)}_{int(args.end/1e6)}_{int(args.bandwidth/1e6)}_{args.fftsize}_{args.frames}_{ts}.bin')
        rows = scan_frequency_range(args.sdr_start, args.sdr_end, args.bandwidth, args.samplerate, args.fftsize, args.frames, output_file)
        frequencies = np.linspace(args.start / 1e6, args.end / 1e6, len(rows))

        try:
            cutoff = args.cutoff
            if not args.cutoff:
                cutoff = np.mean(rows) + np.std(rows) * 2
                print("Cutoff:", args.cutoff)

            peaks, properties = find_peaks(rows, height=cutoff, width=args.width, distance=args.distance)
            peak_freqs = np.linspace(args.start / 1e6, args.end / 1e6, len(rows))[peaks]
            peak_heights = properties['peak_heights']

            for i in range(len(peak_freqs)):
                print(f"Peak at {peak_freqs[i]:.2f} MHz, height {peak_heights[i]:.2f}")

            line.set_data(frequencies, rows)
            line.set_data(frequencies, rows)
            scatter.set_data(peak_freqs, peak_heights)
            ax.set_ylim([min(rows), max(rows) + 10])
            ax.relim()
            fig.canvas.draw()
            fig.canvas.flush_events()
        except Exception as e:
            print("Failed to update plot:", e)
            traceback.print_exc()

def on_click(event):
    global pause, button
    pause = not pause
    button.label.set_text('Resume' if pause else 'Pause')
    plt.draw()

def MHz_to_Hz(value):
    return int(float(value) * 1e6)

def main():
    global pause, button, fig
    signal.signal(signal.SIGINT, signal_handler)

    parser = ArgumentParser(description="FFT Frequency Scanner and Logger")
    parser.add_argument("--dir", type=str, default="./fft", help="Directory to save FFT files")
    parser.add_argument("--bandwidth", type=MHz_to_Hz, default=10, help="Bandwidth per FFT in MHz")
    #parser.add_argument("--samplerate", type=MHz_to_Hz, default=50, help="Sample rate in Hz") # plutosdr has quadrature sampling
    parser.add_argument("--start", type=MHz_to_Hz, default=90, help="Start Frequency in MHz")
    parser.add_argument("--end", type=MHz_to_Hz, default=150, help="End Frequency in MHz")
    parser.add_argument("--fftsize", type=int, default=1024, help="FFT size, must be a power of 2")
    parser.add_argument("--frames", type=int, default=2, help="Number of FFT frames to capture each sample period")
    parser.add_argument("--cutoff", type=float, default=False, help="Cutoff frequency for peaks (default: 2 standard deviations above mean)")
    parser.add_argument("--width", type=int, default=5, help="Minimum width for peak detection")
    parser.add_argument("--distance", type=int, default=30, help="Minimum distance between peaks")
    args = parser.parse_args()

    adj = adjust_bandwidth(args.start, args.end, args.bandwidth, 52e6)
    if adj != args.bandwidth:
        print("Adjusted bandwidth:", adj/1e6, "MHz")
        args.bandwidth = adj

    args.samplerate = args.bandwidth # plutosdr has quadrature sampling, so we can sample at the bandwidth
    args.sdr_start = int(args.start + args.bandwidth / 2)
    args.sdr_end = int(args.end - args.bandwidth / 2)

    plt.ion()

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    ax_button = plt.axes([0.8, 0.05, 0.1, 0.075])
    button = plt.Button(ax_button, 'Pause')
    button.on_clicked(on_click)

    fig.set_size_inches(12, 5)
    line, = ax.plot([0], [0], label='Spectrum')
    scatter, = ax.plot([0], [0], 'x', color='red', label='Detected Peaks')
    ax.set_title('Peak Detection')
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.set_xlim([args.start / 1e6, args.end / 1e6])

    fig.canvas.mpl_connect('key_press_event', pause)

    while True:
        update_data(args, line, scatter, ax, fig)
        plt.pause(0.1)
        time.sleep(0.1)

if __name__ == '__main__':
    main()
