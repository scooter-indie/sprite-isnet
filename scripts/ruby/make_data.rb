#!/usr/bin/env ruby
# frozen_string_literal: true

# ===============================================
# PNG Transparency to Black/White with Rotations
# Non-transparent pixels -> White (255,255,255)
# Transparent pixels -> Black (0,0,0)
# ===============================================

require 'fileutils'
require 'optparse'

# Check if ImageMagick is available
def check_imagemagick
  result = system('magick --version > nul 2>&1')
  unless result
    puts "ERROR: ImageMagick not found!"
    puts "Please install ImageMagick and ensure 'magick' is in your PATH"
    exit 1
  end
end

# Convert PNG transparency to black/white
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
def rotate_image(input_file, output_file, angle)
  if angle == 0
    # Just copy for 0 degrees
    FileUtils.cp(input_file, output_file)
    return true
  elsif angle == 45
    # 45 degrees needs black background fill
    cmd = [
      'magick',
      input_file,
      '-background', 'black',
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

# Process a single file
def process_file(input_file, output_dir)
  unless File.exist?(input_file)
    puts "  ERROR: File not found - #{input_file}"
    return false
  end

  # Get filename without extension
  base_name = File.basename(input_file, '.*')
  
  puts "Processing: #{File.basename(input_file)}"
  
  # Create temporary file for base conversion
  temp_file = File.join(ENV['TEMP'] || '/tmp', "bw_temp_#{rand(10000)}.png")
  
  # Convert transparency to black/white
  unless convert_transparency_to_bw(input_file, temp_file)
    puts "  ERROR: Conversion failed"
    File.delete(temp_file) if File.exist?(temp_file)
    return false
  end
  
  # Create all 5 rotations
  angles = [0, 45, 90, 180, 270]
  success = true
  
  angles.each do |angle|
    output_file = File.join(output_dir, "#{base_name}_#{angle}deg.png")
    print "  [#{angle} deg] Creating: #{File.basename(output_file)} ... "
    
    if rotate_image(temp_file, output_file, angle)
      puts "SUCCESS"
    else
      puts "FAILED"
      success = false
    end
  end
  
  # Cleanup temp file
  File.delete(temp_file) if File.exist?(temp_file)
  
  puts "  #{success ? 'All rotations completed successfully' : 'Some rotations failed'}"
  puts
  
  success
end

# Parse command line options
def parse_options
  options = {}
  
  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby #{File.basename(__FILE__)} --input-dir DIR --output-dir DIR [options]"
    opts.separator ""
    opts.separator "Required options:"
    
    opts.on("--input-dir DIR", "Directory containing source PNG images") do |dir|
      options[:input_dir] = dir
    end
    
    opts.on("--output-dir DIR", "Directory to store processed images") do |dir|
      options[:output_dir] = dir
    end
    
    opts.separator ""
    opts.separator "Optional options:"
    
    opts.on("--pattern PATTERN", "File pattern to match (default: *.png)") do |pattern|
      options[:pattern] = pattern
    end
    
    opts.on("--recursive", "Process subdirectories recursively") do
      options[:recursive] = true
    end
    
    opts.on("-h", "--help", "Show this help message") do
      puts opts
      exit
    end
    
    opts.separator ""
    opts.separator "Examples:"
    opts.separator "  ruby #{File.basename(__FILE__)} --input-dir ./source --output-dir ./processed"
    opts.separator "  ruby #{File.basename(__FILE__)} --input-dir C:\\Images --output-dir C:\\Output"
    opts.separator "  ruby #{File.basename(__FILE__)} --input-dir ./src --output-dir ./dst --pattern sprite_*.png"
    opts.separator "  ruby #{File.basename(__FILE__)} --input-dir ./src --output-dir ./dst --recursive"
    opts.separator ""
    opts.separator "Output:"
    opts.separator "  Creates 5 rotated versions for each input PNG file:"
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
  
  # Set default pattern if not specified
  options[:pattern] ||= "*.png"
  
  # Validate required options
  if options[:input_dir].nil? || options[:output_dir].nil?
    puts "ERROR: Both --input-dir and --output-dir are required!"
    puts
    puts parser
    exit 1
  end
  
  options
end

# Get list of PNG files to process
def get_input_files(input_dir, pattern, recursive)
  unless Dir.exist?(input_dir)
    puts "ERROR: Input directory does not exist: #{input_dir}"
    exit 1
  end
  
  if recursive
    # Recursively find all PNG files
    search_pattern = File.join(input_dir, "**", pattern)
  else
    # Only top-level directory
    search_pattern = File.join(input_dir, pattern)
  end
  
  files = Dir.glob(search_pattern).select { |f| File.file?(f) }
  
  if files.empty?
    puts "WARNING: No files found matching pattern '#{pattern}' in #{input_dir}"
    if recursive
      puts "         (searched recursively)"
    end
  end
  
  files.sort
end

# Main script
def main
  options = parse_options
  
  check_imagemagick
  
  input_dir = File.expand_path(options[:input_dir])
  output_dir = File.expand_path(options[:output_dir])
  pattern = options[:pattern]
  recursive = options[:recursive] || false
  
  # Create output directory if it doesn't exist
  unless Dir.exist?(output_dir)
    puts "Creating output directory: #{output_dir}"
    FileUtils.mkdir_p(output_dir)
  end
  
  # Get list of files to process
  input_files = get_input_files(input_dir, pattern, recursive)
  
  if input_files.empty?
    puts "No files to process. Exiting."
    exit 0
  end
  
  puts "=" * 60
  puts "BATCH PROCESSING: PNG to BW with Rotations"
  puts "=" * 60
  puts "Input directory:  #{input_dir}"
  puts "Output directory: #{output_dir}"
  puts "Pattern:          #{pattern}"
  puts "Recursive:        #{recursive ? 'Yes' : 'No'}"
  puts "Files found:      #{input_files.length}"
  puts "=" * 60
  puts
  
  total = 0
  success = 0
  failed = 0
  
  input_files.each do |file|
    total += 1
    puts "[#{total}/#{input_files.length}]"
    if process_file(file, output_dir)
      success += 1
    else
      failed += 1
    end
  end
  
  puts "=" * 60
  puts "BATCH PROCESSING COMPLETE"
  puts "=" * 60
  puts "Total files processed: #{total}"
  puts "Successful:            #{success}"
  puts "Failed:                #{failed}"
  puts "Output directory:      #{output_dir}"
  puts "=" * 60
  
  exit(failed > 0 ? 1 : 0)
end

main if __FILE__ == $PROGRAM_NAME
