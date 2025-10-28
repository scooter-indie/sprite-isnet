#!/usr/bin/env ruby
# frozen_string_literal: true

# ===============================================
# PNG Processing with Rotation and Masking
# 
# Processes raw and background-removed spritesheets:
# - Renames matching files to timestamp-based names
# - Rotates raw files (preserves pixels)
# - Masks and rotates BGR files (transparency to B/W)
# ===============================================

require 'fileutils'
require 'optparse'
require 'time'

# Check if ImageMagick is available
def check_imagemagick
  result = system('magick --version > nul 2>&1')
  unless result
    puts "ERROR: ImageMagick not found!"
    puts "Please install ImageMagick and ensure 'magick' is in your PATH"
    exit 1
  end
end

# Convert PNG transparency to black/white (masking)
def convert_transparency_to_bw(input_file, output_file)
  # Extract alpha: opaque=white(255), transparent=black(0)
  cmd = [
    'magick', input_file,
    '-alpha', 'extract',
    output_file
  ].join(' ')
  
  system(cmd)
end

# Rotate image
def rotate_image(input_file, output_file, angle, keep_pixels: false)
  if angle == 0
    # Just copy for 0 degrees
    FileUtils.cp(input_file, output_file)
    return true
  elsif angle == 45
    # 45 degrees needs background fill
    # Use transparent background if keeping pixels, black if converting
    bg_color = keep_pixels ? 'none' : 'black'
    cmd = [
      'magick',
      input_file,
      '-background', bg_color,
      '-rotate', angle.to_s,
      output_file
    ].join(' ')
  else
    # 90, 180, 270 degrees
    cmd = [
      'magick',
      input_file,
      '-rotate', angle.to_s,
      output_file
    ].join(' ')
  end
  
  system(cmd)
end

# Remove "-nobg-fuzzy" from filename
def clean_filename(filename)
  # Remove -nobg-fuzzy before the extension
  filename.sub(/-nobg-fuzzy(?=\.[^.]+$)/i, '')
end

# Generate timestamp-based filename
def generate_timestamp_filename(counter)
  timestamp = Time.now.strftime("%Y%m%d_%H%M%S")
  "#{timestamp}_#{counter.to_s.rjust(3, '0')}.png"
end

# Get list of PNG files in directory
def get_png_files(directory)
  unless Dir.exist?(directory)
    puts "ERROR: Directory does not exist: #{directory}"
    exit 1
  end
  
  Dir.glob(File.join(directory, "*.png")).select { |f| File.file?(f) }.sort
end

# Rename files in BGR directory (remove -nobg-fuzzy)
def clean_bgr_filenames(in_bgr)
  puts "Cleaning filenames in BGR directory..."
  
  files = get_png_files(in_bgr)
  renamed_count = 0
  
  files.each do |file|
    original_name = File.basename(file)
    cleaned_name = clean_filename(original_name)
    
    if original_name != cleaned_name
      new_path = File.join(in_bgr, cleaned_name)
      
      if File.exist?(new_path)
        puts "  WARNING: Cannot rename '#{original_name}' to '#{cleaned_name}' - target already exists"
      else
        File.rename(file, new_path)
        puts "  Renamed: #{original_name} -> #{cleaned_name}"
        renamed_count += 1
      end
    end
  end
  
  puts "  Cleaned #{renamed_count} filename(s)"
  puts
end

# Match files between raw and BGR directories
def match_files(in_raw, in_bgr)
  raw_files = get_png_files(in_raw).map { |f| File.basename(f) }
  bgr_files = get_png_files(in_bgr).map { |f| File.basename(f) }
  
  raw_set = raw_files.to_set
  bgr_set = bgr_files.to_set
  
  matched = raw_set & bgr_set
  unmatched_raw = raw_set - bgr_set
  unmatched_bgr = bgr_set - raw_set
  
  {
    matched: matched.to_a.sort,
    unmatched_raw: unmatched_raw.to_a.sort,
    unmatched_bgr: unmatched_bgr.to_a.sort
  }
end

# Rename matching files to timestamp-based names
def rename_matched_files(in_raw, in_bgr, matched_files)
  puts "Renaming matched files to timestamp-based names..."
  
  counter = 1
  rename_map = {}
  
  matched_files.each do |filename|
    new_name = generate_timestamp_filename(counter)
    
    raw_old_path = File.join(in_raw, filename)
    raw_new_path = File.join(in_raw, new_name)
    
    bgr_old_path = File.join(in_bgr, filename)
    bgr_new_path = File.join(in_bgr, new_name)
    
    # Check if target names already exist
    if File.exist?(raw_new_path) || File.exist?(bgr_new_path)
      puts "  ERROR: Target filename already exists: #{new_name}"
      next
    end
    
    # Rename both files
    File.rename(raw_old_path, raw_new_path)
    File.rename(bgr_old_path, bgr_new_path)
    
    rename_map[new_name] = filename
    puts "  [#{counter}] #{filename} -> #{new_name}"
    
    counter += 1
    
    # Small delay to ensure unique timestamps if processing very quickly
    sleep(0.001)
  end
  
  puts "  Renamed #{rename_map.size} file pair(s)"
  puts
  
  rename_map
end

# Process raw file (rotation only, keep pixels)
def process_raw_file(input_file, output_dir)
  base_name = File.basename(input_file, '.*')
  
  angles = [0, 45, 90, 180, 270]
  success = true
  
  angles.each do |angle|
    output_file = File.join(output_dir, "#{base_name}_#{angle}deg.png")
    
    unless rotate_image(input_file, output_file, angle, keep_pixels: true)
      puts "    ERROR: Failed to rotate #{File.basename(input_file)} at #{angle} degrees"
      success = false
    end
  end
  
  success
end

# Process BGR file (masking + rotation)
def process_bgr_file(input_file, output_dir)
  base_name = File.basename(input_file, '.*')
  
  # Create temporary file for masking
  temp_file = File.join(ENV['TEMP'] || '/tmp', "mask_temp_#{rand(10000)}.png")
  
  # Apply masking (transparency to B/W)
  unless convert_transparency_to_bw(input_file, temp_file)
    puts "    ERROR: Masking failed for #{File.basename(input_file)}"
    File.delete(temp_file) if File.exist?(temp_file)
    return false
  end
  
  # Create rotations
  angles = [0, 45, 90, 180, 270]
  success = true
  
  angles.each do |angle|
    output_file = File.join(output_dir, "#{base_name}_#{angle}deg.png")
    
    unless rotate_image(temp_file, output_file, angle, keep_pixels: false)
      puts "    ERROR: Failed to rotate #{File.basename(input_file)} at #{angle} degrees"
      success = false
    end
  end
  
  # Cleanup temp file
  File.delete(temp_file) if File.exist?(temp_file)
  
  success
end

# Process all files
def process_files(in_raw, in_bgr, out_original, out_masked)
  raw_files = get_png_files(in_raw)
  bgr_files = get_png_files(in_bgr)
  
  total = raw_files.length
  success_count = 0
  failed_count = 0
  
  puts "Processing #{total} file pair(s)..."
  puts
  
  raw_files.each_with_index do |raw_file, index|
    base_name = File.basename(raw_file)
    bgr_file = File.join(in_bgr, base_name)
    
    puts "[#{index + 1}/#{total}] Processing: #{base_name}"
    
    # Process raw file
    print "  RAW -> Rotating (keeping pixels)... "
    if process_raw_file(raw_file, out_original)
      puts "SUCCESS"
      raw_success = true
    else
      puts "FAILED"
      raw_success = false
    end
    
    # Process BGR file
    print "  BGR -> Masking and rotating... "
    if File.exist?(bgr_file)
      if process_bgr_file(bgr_file, out_masked)
        puts "SUCCESS"
        bgr_success = true
      else
        puts "FAILED"
        bgr_success = false
      end
    else
      puts "SKIPPED (no matching BGR file)"
      bgr_success = false
    end
    
    if raw_success && bgr_success
      success_count += 1
    else
      failed_count += 1
    end
    
    puts
  end
  
  { success: success_count, failed: failed_count, total: total }
end

# Parse command line options
def parse_options
  options = {}
  
  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby #{File.basename(__FILE__)} --in-raw DIR --in-bgr DIR --out-original DIR --out-masked DIR"
    opts.separator ""
    opts.separator "Required options:"
    
    opts.on("--in-raw DIR", "Input directory containing raw spritesheets") do |dir|
      options[:in_raw] = dir
    end
    
    opts.on("--in-bgr DIR", "Input directory containing background-removed spritesheets") do |dir|
      options[:in_bgr] = dir
    end
    
    opts.on("--out-original DIR", "Output directory for rotated raw spritesheets") do |dir|
      options[:out_original] = dir
    end
    
    opts.on("--out-masked DIR", "Output directory for masked and rotated spritesheets") do |dir|
      options[:out_masked] = dir
    end
    
    opts.separator ""
    opts.separator "Optional options:"
    
    opts.on("-h", "--help", "Show this help message") do
      puts opts
      exit
    end
    
    opts.separator ""
    opts.separator "Examples:"
    opts.separator "  ruby #{File.basename(__FILE__)} --in-raw ./raw --in-bgr ./bgr --out-original ./original --out-masked ./masked"
    opts.separator "  ruby #{File.basename(__FILE__)} --in-raw C:\\raw --in-bgr C:\\bgr --out-original C:\\original --out-masked C:\\masked"
    opts.separator ""
    opts.separator "Processing Steps:"
    opts.separator "  1. Clean BGR filenames (remove '-nobg-fuzzy' suffix)"
    opts.separator "  2. Match files between raw and BGR directories"
    opts.separator "  3. Rename matched files to timestamp-based names"
    opts.separator "  4. Process raw files: rotate only (5 angles) -> out-original"
    opts.separator "  5. Process BGR files: mask + rotate (5 angles) -> out-masked"
    opts.separator ""
    opts.separator "Output:"
    opts.separator "  Creates 5 rotated versions for each input file:"
    opts.separator "    - filename_0deg.png   (no rotation)"
    opts.separator "    - filename_45deg.png  (45 degrees clockwise)"
    opts.separator "    - filename_90deg.png  (90 degrees clockwise)"
    opts.separator "    - filename_180deg.png (180 degrees)"
    opts.separator "    - filename_270deg.png (270 degrees clockwise)"
  end
  
  begin
    parser.parse!
  rescue OptionParser::InvalidOption, OptionParser::MissingArgument => e
    puts "ERROR: #{e.message}"
    puts
    puts parser
    exit 1
  end
  
  # Validate required options
  required = [:in_raw, :in_bgr, :out_original, :out_masked]
  missing = required.select { |opt| options[opt].nil? }
  
  unless missing.empty?
    puts "ERROR: Missing required option(s): #{missing.map { |o| "--#{o.to_s.tr('_', '-')}" }.join(', ')}"
    puts
    puts parser
    exit 1
  end
  
  options
end

# Main script
def main
  options = parse_options
  
  check_imagemagick
  
  in_raw = File.expand_path(options[:in_raw])
  in_bgr = File.expand_path(options[:in_bgr])
  out_original = File.expand_path(options[:out_original])
  out_masked = File.expand_path(options[:out_masked])
  
  # Verify input directories exist
  [in_raw, in_bgr].each do |dir|
    unless Dir.exist?(dir)
      puts "ERROR: Input directory does not exist: #{dir}"
      exit 1
    end
  end
  
  # Create output directories if they don't exist
  [out_original, out_masked].each do |dir|
    unless Dir.exist?(dir)
      puts "Creating output directory: #{dir}"
      FileUtils.mkdir_p(dir)
    end
  end
  
  puts "=" * 70
  puts "PNG PROCESSING: Rotation and Masking"
  puts "=" * 70
  puts "Input RAW directory:  #{in_raw}"
  puts "Input BGR directory:  #{in_bgr}"
  puts "Output Original dir:  #{out_original}"
  puts "Output Masked dir:    #{out_masked}"
  puts "=" * 70
  puts
  
  # Step 1: Clean BGR filenames
  clean_bgr_filenames(in_bgr)
  
  # Step 2: Match files
  puts "Matching files between RAW and BGR directories..."
  match_result = match_files(in_raw, in_bgr)
  
  puts "  Matched files:        #{match_result[:matched].length}"
  puts "  Unmatched in RAW:     #{match_result[:unmatched_raw].length}"
  puts "  Unmatched in BGR:     #{match_result[:unmatched_bgr].length}"
  
  if match_result[:unmatched_raw].any?
    puts
    puts "  Files in RAW without BGR match:"
    match_result[:unmatched_raw].each do |filename|
      puts "    - #{filename}"
    end
  end
  
  if match_result[:unmatched_bgr].any?
    puts
    puts "  Files in BGR without RAW match:"
    match_result[:unmatched_bgr].each do |filename|
      puts "    - #{filename}"
    end
  end
  puts
  
  if match_result[:matched].empty?
    puts "ERROR: No matching files found between RAW and BGR directories"
    exit 1
  end
  
  # Step 3: Rename matched files
  rename_map = rename_matched_files(in_raw, in_bgr, match_result[:matched])
  
  # Step 4 & 5: Process files
  results = process_files(in_raw, in_bgr, out_original, out_masked)
  
  puts "=" * 70
  puts "PROCESSING COMPLETE"
  puts "=" * 70
  puts "Total file pairs processed: #{results[:total]}"
  puts "Successful:                 #{results[:success]}"
  puts "Failed:                     #{results[:failed]}"
  puts "Output directories:"
  puts "  Original (rotated):       #{out_original}"
  puts "  Masked (B/W + rotated):   #{out_masked}"
  puts "=" * 70
  
  exit(results[:failed] > 0 ? 1 : 0)
end

main if __FILE__ == $PROGRAM_NAME
