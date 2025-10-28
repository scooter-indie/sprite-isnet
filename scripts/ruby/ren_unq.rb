#!/usr/bin/env ruby
# rename_with_timestamp.rb
# Renames all .png files in a directory to YYYYMMDD_HHMMSS_NNN.png format

require 'optparse'
require 'fileutils'
require 'time'

def parse_options
  options = {}
  
  parser = OptionParser.new do |opts|
    opts.banner = "Usage: ruby #{File.basename(__FILE__)} --dir DIRECTORY"
    opts.separator ""
    opts.separator "Required options:"
    
    opts.on("--dir DIRECTORY", "Target directory containing .png files to rename") do |dir|
      options[:dir] = dir
    end
    
    opts.separator ""
    opts.on("-h", "--help", "Show this help message") do
      puts opts
      exit
    end
    
    opts.separator ""
    opts.separator "Description:"
    opts.separator "  Renames all .png files in the specified directory to:"
    opts.separator "  YYYYMMDD_HHMMSS_NNN.png (e.g., 20251028_133045_001.png)"
    opts.separator ""
    opts.separator "  - YYYYMMDD: Current date"
    opts.separator "  - HHMMSS: Current time"
    opts.separator "  - NNN: Sequential counter (001, 002, 003, etc.)"
    opts.separator ""
    opts.separator "  Only processes .png files in the target directory (non-recursive)."
    opts.separator "  No duplicate filenames will be created."
    opts.separator ""
    opts.separator "Example:"
    opts.separator "  ruby #{File.basename(__FILE__)} --dir C:\\sprites\\raw"
    opts.separator "  ruby #{File.basename(__FILE__)} --dir ./images"
  end
  
  begin
    parser.parse!
  rescue OptionParser::InvalidOption, OptionParser::MissingArgument => e
    puts "ERROR: #{e.message}"
    puts ""
    puts parser
    exit 1
  end
  
  # Validate required option
  if options[:dir].nil?
    puts "ERROR: Missing required option: --dir"
    puts ""
    puts parser
    exit 1
  end
  
  options
end

def get_png_files(directory)
  Dir.glob(File.join(directory, "*.png")).select { |f| File.file?(f) }
end

def generate_timestamp
  Time.now.strftime("%Y%m%d_%H%M%S")
end

def rename_files(directory)
  png_files = get_png_files(directory)
  
  if png_files.empty?
    puts "No .png files found in directory: #{directory}"
    return
  end
  
  puts "Found #{png_files.length} .png file(s) to rename"
  puts ""
  
  # Generate timestamp once for all files
  timestamp = generate_timestamp
  counter = 1
  success_count = 0
  failed_count = 0
  
  png_files.each do |old_path|
    old_name = File.basename(old_path)
    
    # Generate new filename with sequential counter
    new_name = sprintf("%s_%03d.png", timestamp, counter)
    new_path = File.join(directory, new_name)
    
    # Check if new filename already exists (shouldn't happen, but safety check)
    while File.exist?(new_path)
      counter += 1
      new_name = sprintf("%s_%03d.png", timestamp, counter)
      new_path = File.join(directory, new_name)
    end
    
    begin
      File.rename(old_path, new_path)
      puts "[#{counter}] #{old_name} -> #{new_name}"
      success_count += 1
      counter += 1
    rescue => e
      puts "[#{counter}] FAILED: #{old_name} (#{e.message})"
      failed_count += 1
      counter += 1
    end
  end
  
  puts ""
  puts "=" * 60
  puts "Summary:"
  puts "  Successfully renamed: #{success_count}"
  puts "  Failed: #{failed_count}"
  puts "  Total: #{png_files.length}"
  puts "=" * 60
end

def main
  options = parse_options
  
  directory = File.expand_path(options[:dir])
  
  # Verify directory exists
  unless Dir.exist?(directory)
    puts "ERROR: Directory does not exist: #{directory}"
    exit 1
  end
  
  puts "Target directory: #{directory}"
  puts ""
  
  rename_files(directory)
end

# Run script
main if __FILE__ == $PROGRAM_NAME
