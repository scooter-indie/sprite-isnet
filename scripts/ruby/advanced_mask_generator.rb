#!/usr/bin/env ruby
# advanced_mask_generator.rb - Multiple mask generation methods

require 'fileutils'
require 'json'

class AdvancedMaskGenerator
  def initialize(input_dir, output_images_dir, output_masks_dir, options = {})
    @input_dir = input_dir
    @output_images_dir = output_images_dir
    @output_masks_dir = output_masks_dir
    
    # Options
    @method = options[:method] || 'auto'
    @fuzz = options[:fuzz] || 10
    @preview = options[:preview] || false
    @verbose = options[:verbose] || false
    
    # Create directories
    FileUtils.mkdir_p(@output_images_dir)
    FileUtils.mkdir_p(@output_masks_dir)
    
    if @preview
      @preview_dir = File.join(File.dirname(@output_masks_dir), 'previews')
      FileUtils.mkdir_p(@preview_dir)
    end
  end
  
  def process_all
    image_files = Dir.glob(File.join(@input_dir, '*.{png,jpg,jpeg,PNG,JPG,JPEG}'))
    
    if image_files.empty?
      puts "✗ No images found in #{@input_dir}"
      return
    end
    
    puts "╔═══════════════════════════════════════════════════════════════╗"
    puts "║         Advanced Sprite Mask Generator                       ║"
    puts "╚═══════════════════════════════════════════════════════════════╝"
    puts ""
    puts "Found: #{image_files.length} images"
    puts "Method: #{@method}"
    puts "Fuzz: #{@fuzz}%"
    puts "Preview: #{@preview}"
    puts "=" * 60
    
    success_count = 0
    fail_count = 0
    stats = []
    
    image_files.each_with_index do |input_path, idx|
      basename = File.basename(input_path)
      print "[#{idx + 1}/#{image_files.length}] #{basename}... "
      
      begin
        result = process_single_image(input_path)
        puts "✓ (#{result[:coverage].round(1)}% coverage)"
        success_count += 1
        stats << result
      rescue => e
        puts "✗ Error: #{e.message}"
        fail_count += 1
      end
    end
    
    print_summary(success_count, fail_count, stats)
  end
  
  def process_single_image(input_path)
    basename = File.basename(input_path, File.extname(input_path))
    
    # Copy image
    output_image = File.join(@output_images_dir, "#{basename}.png")
    system("magick", "convert", input_path, "-strip", output_image)
    
    # Generate mask based on method
    output_mask = File.join(@output_masks_dir, "#{basename}.png")
    
    case @method
    when 'corner'
      generate_mask_corner_sample(input_path, output_mask)
    when 'edge'
      generate_mask_edge_detection(input_path, output_mask)
    when 'threshold'
      generate_mask_threshold(input_path, output_mask)
    when 'auto'
      generate_mask_auto(input_path, output_mask)
    else
      raise "Unknown method: #{@method}"
    end
    
    # Calculate coverage
    coverage = calculate_mask_coverage(output_mask)
    
    # Create preview if requested
    if @preview
      create_preview(output_image, output_mask, basename)
    end
    
    { file: basename, coverage: coverage, method: @method }
  end
  
  def generate_mask_corner_sample(input_path, output_path)
    """
    Method 1: Sample color from corner
    - Best for: Solid color backgrounds
    - Fast and simple
    """
    
    temp_file = "#{output_path}.temp.png"
    
    # Step 1: Select background color (from corner) and make it transparent
    system("magick", input_path,
           "-fuzz", "#{@fuzz}%",
           "-fill", "none",
           "-draw", "color 0,0 floodfill",
           temp_file)
    
    # Step 2: Create binary mask (transparent = black, opaque = white)
    system("magick", temp_file,
           "-alpha", "extract",
           "-negate",
           output_path)
    
    File.delete(temp_file) if File.exist?(temp_file)
  end
  
  def generate_mask_edge_detection(input_path, output_path)
    """
    Method 2: Edge detection + morphology
    - Best for: Complex backgrounds with gradients
    - More robust but slower
    """
    
    temp_file = "#{output_path}.temp.png"
    
    # Step 1: Edge detection
    system("magick", input_path,
           "-colorspace", "Gray",
           "-edge", "1",
           "-threshold", "15%",
           temp_file)
    
    # Step 2: Morphological operations to close gaps
    system("magick", temp_file,
           "-morphology", "Close", "Diamond:3",
           "-negate",
           output_path)
    
    File.delete(temp_file) if File.exist?(temp_file)
  end
  
  def generate_mask_threshold(input_path, output_path)
    """
    Method 3: Luminosity threshold
    - Best for: Backgrounds significantly darker/lighter than sprites
    - Simple but effective for high contrast
    """
    
    system("magick", input_path,
           "-colorspace", "Gray",
           "-threshold", "50%",
           output_path)
  end
  
  def generate_mask_auto(input_path, output_path)
    """
    Method 4: Automatic - tries multiple methods and picks best
    """
    
    # Analyze image to determine best method
    bg_color = get_background_color(input_path)
    
    if bg_color[:is_solid]
      puts "(using corner method)" if @verbose
      generate_mask_corner_sample(input_path, output_path)
    else
      puts "(using edge method)" if @verbose
      generate_mask_edge_detection(input_path, output_path)
    end
  end
  
  def get_background_color(input_path)
    """Analyze background color uniformity"""
    
    # Sample 4 corners and check if similar
    corners = ['0,0', '0,100%', '100%,0', '100%,100%']
    colors = []
    
    corners.each do |coord|
      result = `magick identify -format "%[pixel:p{#{coord}}]" "#{input_path}"`.strip
      colors << result
    end
    
    # Check if all corners are similar (within tolerance)
    is_solid = colors.uniq.length <= 2
    
    { is_solid: is_solid, color: colors.first }
  end
  
  def calculate_mask_coverage(mask_path)
    """Calculate percentage of white pixels in mask"""
    
    result = `magick "#{mask_path}" -format "%[fx:mean*100]" info:`.strip
    result.to_f
  end
  
  def create_preview(image_path, mask_path, basename)
    """Create side-by-side preview with overlay"""
    
    preview_path = File.join(@preview_dir, "#{basename}_preview.png")
    overlay_path = "#{preview_path}.overlay.png"
    
    # Create colored overlay (mask in green)
    system("magick", image_path,
           "(", mask_path, "-colorspace", "RGB", "-fill", "lime", "-colorize", "100", ")",
           "-compose", "blend",
           "-define", "compose:args=70,30",
           "-composite",
           overlay_path)
    
    # Create side-by-side comparison
    system("magick", 
           image_path,
           mask_path,
           overlay_path,
           "+append",
           "-gravity", "North",
           "-pointsize", "24",
           "-fill", "white",
           "-stroke", "black",
           "-strokewidth", "2",
           "-annotate", "+10+10", "Original",
           "-stroke", "none",
           "-fill", "white",
           "-annotate", "+10+10", "Original",
           preview_path)
    
    File.delete(overlay_path) if File.exist?(overlay_path)
  end
  
  def print_summary(success, fail, stats)
    puts "=" * 60
    puts "Processing Complete!"
    puts "=" * 60
    puts "Successful: #{success}"
    puts "Failed: #{fail}"
    
    if stats.length > 0
      coverages = stats.map { |s| s[:coverage] }
      avg_coverage = coverages.sum / coverages.length
      min_coverage = coverages.min
      max_coverage = coverages.max
      
      puts ""
      puts "Mask Coverage Statistics:"
      puts "  Average: #{avg_coverage.round(1)}%"
      puts "  Minimum: #{min_coverage.round(1)}%"
      puts "  Maximum: #{max_coverage.round(1)}%"
      
      # Warn about unusual coverages
      outliers = stats.select { |s| s[:coverage] < 5 || s[:coverage] > 95 }
      if outliers.length > 0
        puts ""
        puts "⚠ Warning: #{outliers.length} masks with unusual coverage:"
        outliers.each do |stat|
          puts "  - #{stat[:file]}: #{stat[:coverage].round(1)}%"
        end
      end
    end
    
    puts ""
    puts "Output directories:"
    puts "  Images: #{@output_images_dir}"
    puts "  Masks: #{@output_masks_dir}"
    puts "  Previews: #{@preview_dir}" if @preview
  end
end

# CLI
if __FILE__ == $0
  require 'optparse'
  
  options = {
    method: 'auto',
    fuzz: 10,
    preview: false,
    verbose: false
  }
  
  OptionParser.new do |opts|
    opts.banner = "Usage: ruby advanced_mask_generator.rb [options] <input_dir> <output_images_dir> <output_masks_dir>"
    
    opts.on("-m", "--method METHOD", "Method: auto, corner, edge, threshold (default: auto)") do |v|
      options[:method] = v
    end
    
    opts.on("-f", "--fuzz PERCENT", Integer, "Fuzz percentage for color tolerance (default: 10)") do |v|
      options[:fuzz] = v
    end
    
    opts.on("-p", "--preview", "Generate preview images") do
      options[:preview] = true
    end
    
    opts.on("-v", "--verbose", "Verbose output") do
      options[:verbose] = true
    end
    
    opts.on("-h", "--help", "Show this help") do
      puts opts
      exit
    end
  end.parse!
  
  if ARGV.length < 3
    puts "Error: Missing required arguments"
    puts ""
    puts "Usage: ruby advanced_mask_generator.rb [options] <input_dir> <output_images_dir> <output_masks_dir>"
    puts ""
    puts "Example:"
    puts "  ruby advanced_mask_generator.rb -m auto -f 15 -p C:\\sprite-data\\raw C:\\sprite-data\\train\\images C:\\sprite-data\\train\\masks"
    exit 1
  end
  
  input_dir = ARGV[0]
  output_images_dir = ARGV[1]
  output_masks_dir = ARGV[2]
  
  generator = AdvancedMaskGenerator.new(input_dir, output_images_dir, output_masks_dir, options)
  generator.process_all
end
