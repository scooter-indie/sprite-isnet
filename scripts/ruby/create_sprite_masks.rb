#!/usr/bin/env ruby
# create_sprite_masks.rb - Generate binary masks from sprite sheets

require 'fileutils'

class SpriteMaskGenerator
  def initialize(input_dir, output_images_dir, output_masks_dir)
    @input_dir = input_dir
    @output_images_dir = output_images_dir
    @output_masks_dir = output_masks_dir
    
    # Create output directories
    FileUtils.mkdir_p(@output_images_dir)
    FileUtils.mkdir_p(@output_masks_dir)
  end
  
  def process_all
    # Get all image files
    image_files = Dir.glob(File.join(@input_dir, '*.{png,jpg,jpeg,PNG,JPG,JPEG}'))
    
    if image_files.empty?
      puts "✗ No images found in #{@input_dir}"
      return
    end
    
    puts "Found #{image_files.length} images to process"
    puts "=" * 60
    
    success_count = 0
    fail_count = 0
    
    image_files.each_with_index do |input_path, idx|
      print "[#{idx + 1}/#{image_files.length}] Processing #{File.basename(input_path)}... "
      
      begin
        process_single_image(input_path)
        puts "✓"
        success_count += 1
      rescue => e
        puts "✗ Error: #{e.message}"
        fail_count += 1
      end
    end
    
    puts "=" * 60
    puts "Complete! Success: #{success_count}, Failed: #{fail_count}"
  end
  
  def process_single_image(input_path)
    basename = File.basename(input_path, File.extname(input_path))
    
    # Copy/convert image to output directory
    output_image = File.join(@output_images_dir, "#{basename}.png")
    
    # Convert image to PNG if needed
    system("magick", "convert", input_path, output_image)
    
    # Generate mask
    output_mask = File.join(@output_masks_dir, "#{basename}.png")
    generate_mask_color_threshold(input_path, output_mask)
  end
  
  def generate_mask_color_threshold(input_path, output_path, fuzz: 10)
    """
    Generate mask by selecting background color from corner and removing it.
    
    Method 1: Color threshold (works best for solid backgrounds)
    - Samples color from top-left corner
    - Creates mask where similar colors become black, rest becomes white
    """
    
    # Sample background color from top-left corner (pixel 0,0)
    # Then create mask: background = black (0), sprites = white (255)
    
    system("magick", "convert", input_path,
           "(", "+clone", 
           "-crop", "1x1+0+0",
           "-format", "%[pixel:p{0,0}]",
           "info:-",
           ")",
           "-fill", "white",
           "+opaque", "pixel:-",
           "-fill", "black",
           "-opaque", "pixel:-",
           "-fuzz", "#{fuzz}%",
           output_path)
  end
end

# Usage
if __FILE__ == $0
  if ARGV.length < 3
    puts "Usage: ruby create_sprite_masks.rb <input_dir> <output_images_dir> <output_masks_dir>"
    puts ""
    puts "Example:"
    puts "  ruby create_sprite_masks.rb C:\\sprite-data\\raw C:\\sprite-data\\train\\images C:\\sprite-data\\train\\masks"
    exit 1
  end
  
  input_dir = ARGV[0]
  output_images_dir = ARGV[1]
  output_masks_dir = ARGV[2]
  
  generator = SpriteMaskGenerator.new(input_dir, output_images_dir, output_masks_dir)
  generator.process_all
end
