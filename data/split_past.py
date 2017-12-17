import os
import subprocess as sp

def split_vid_from_path(video_file_path, output_file_path, start_time, end_time):
	pipe = sp.Popen(["ffmpeg","-v", "quiet", "-y", "-i", video_file_path, "-vcodec", "copy", "-acodec", "copy",
                 "-ss", start_time, "-to", end_time, "-sn", output_file_path ])
	pipe.wait()
	return True

def main():

	sample_vid = os.path.join('/home/hpc/github/WeatherClassification/data_cctv/ss/', "ss_20170417_1415-1430_day_20.avi")

	for num in range(0,300):

		output_name = 'ss_20170417_1415-1430_day_20_' + '%03d.avi' %num
		output_vid = os.path.join('/home/hpc/github/WeatherClassification/data_cctv/rain_15', output_name)
		start = num * 3
		end = start + 3
		split_vid_from_path(sample_vid, output_vid, str(start), str(end))

if __name__== '__main__':
	main()