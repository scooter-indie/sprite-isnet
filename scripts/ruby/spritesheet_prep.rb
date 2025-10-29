#!/usr/bin/env ruby
# spritesheet_prep.rb
# Replaces transparent pixels in PNG spritesheets with unique dark colors
# NEW: --fixup flag to replace transparency with magenta in *_45deg.png files

require 'optparse'
require 'fileutils'
require 'set'

class SpritesheetPrepper
  SUFFIX = "-processed" # Still used for temp files
  
  # Dark color range: RGB values 0-80 for darker spectrum
  DARK_COLOR_MAX = 80
  
  # Color scheme generators
  COLOR_SCHEMES = {
    magenta: ->(index) {
      # Magenta: Higher red and blue, lower green (more visible)
      base = 10 + (index * 4) % (DARK_COLOR_MAX - 10)
      r = base + 10
      g = [base - 5, 0].max
      b = base + 10
      [r, g, b]
    },
    cyan: ->(index) {
      # Cyan: Higher green and blue, lower red (more visible)
      base = 10 + (index * 4) % (DARK_COLOR_MAX - 10)
      r = [base - 5, 0].max
      g = base + 10
      b = base + 10
      [r, g, b]
    },
    gray: ->(index) {
      # Gray: Equal RGB values
      val = 5 + (index * 3) % (DARK_COLOR_MAX - 5)
      [val, val, val]
    },
    dark: ->(index) {
      # Default: Very dark colors, blue-biased
      r = (index / ((DARK_COLOR_MAX + 1) * (DARK_COLOR_MAX + 1))) % (DARK_COLOR_MAX + 1)
      g = ((index / (DARK_COLOR_MAX + 1)) % (DARK_COLOR_MAX + 1))
      b = (index % (DARK_COLOR_MAX + 1))
      [r, g, b]
    }
  }
  
  def initialize(color_scheme = :dark, fixup_mode = false)
    @color_scheme = color_scheme
    @fixup_mode = fixup_mode
    @stats = {
      files_processed: 0,
      files_skipped: 0,
      files_errored: 0,
      errors: []
    }
  end
  
  def run(options)
    puts "=" * 70
    
    if @fixup_mode
      puts "Spritesheet Fixup Tool (45-degree rotation artifacts)"
      puts "Mode: Replace transparency with MAGENTA"
    else
      puts "Spritesheet Transparency Replacement Tool"
      puts "Color Scheme: #{@color_scheme.to_s.capitalize}"
    end
    
    puts "=" * 70
    puts
    
    # Verify ImageMagick is available
    unless command_exists?("magick")
      abort "ERROR: ImageMagick 'magick' command not found. Please ensure ImageMagick is installed and in PATH."
    end
    
    files = collect_files(options)
    
    if files.empty?
      if @fixup_mode
        puts "No *_45deg.png files found to fix."
      else
        puts "No PNG files found to process."
      end
      return
    end
    
    puts "Found #{files.length} PNG file(s) to process"
    puts
    
    files.each_with_index do |file, index|
      puts "[#{index + 1}/#{files.length}] Processing: #{File.basename(file)}"
      
      if @fixup_mode
        process_fixup_file(file)
      else
        process_file(file)
      end
      
      puts
    end
    
    print_summary
  end
  
  private
  
  def collect_files(options)
    files = []
    
    if options[:image]
      if File.exist?(options[:image])
        if File.extname(options[:image]).downcase == ".png"
          # In fixup mode, verify filename ends with _45deg.png
          if @fixup_mode
            if options[:image] =~ /_45deg\.png$/i
              files << options[:image]
            else
              abort "ERROR: In fixup mode, file must end with _45deg.png: #{options[:image]}"
            end
          else
            files << options[:image]
          end
        else
          abort "ERROR: File must be a PNG image: #{options[:image]}"
        end
      else
        abort "ERROR: File not found: #{options[:image]}"
      end
      
    elsif options[:dir]
      if Dir.exist?(options[:dir])
        if @fixup_mode
          # Only collect *_45deg.png files
          files = Dir.glob(File.join(options[:dir], "*_45deg.png"))
        else
          files = Dir.glob(File.join(options[:dir], "*.png"))
          # No longer skip processed files since we overwrite originals
        end
      else
        abort "ERROR: Directory not found: #{options[:dir]}"
      end
    end
    
    files
  end
  
  # NEW: Process file in fixup mode (replace transparency with magenta)
  def process_fixup_file(input_file)
    temp_output = nil
    
    begin
      # Check if file is valid PNG
      unless valid_png?(input_file)
        raise "File is not a valid PNG image"
      end
      
      # Check if image has transparency
      unless has_transparency?(input_file)
        puts "  ⚠ SKIPPED: No transparent pixels found"
        @stats[:files_skipped] += 1
        return
      end
      
      # Create temporary output filename
      temp_output = generate_temp_filename(input_file)
      
      # Replace transparent pixels with magenta
      puts "  → Replacing transparent pixels with MAGENTA..."
      replace_transparency_with_magenta(input_file, temp_output)
      
      # Verify temp output was created
      unless File.exist?(temp_output)
        raise "Temporary output file was not created"
      end
      
      # Replace original with processed version
      puts "  → Saving to original filename..."
      FileUtils.mv(temp_output, input_file)
      temp_output = nil # Clear since we've moved it
      
      puts "  ✓ SUCCESS: Fixed #{File.basename(input_file)}"
      @stats[:files_processed] += 1
      
    rescue => e
      puts "  ✗ ERROR: #{e.message}"
      @stats[:files_errored] += 1
      @stats[:errors] << { file: File.basename(input_file), error: e.message }
      
      # Clean up temp file if it exists
      File.delete(temp_output) if temp_output && File.exist?(temp_output)
    end
  end
  
  # NEW: Replace fully transparent pixels with magenta (technique from make_data.rb)
  def replace_transparency_with_magenta(input_file, output_file)
    # Using the technique from make_data.rb for rotation background fill
    # -background magenta: Fill color for transparent areas
    # -alpha remove: Flatten alpha channel using background color
    exec_magick("magick \"#{input_file}\" -background magenta -alpha remove \"#{output_file}\"")
  end
  
  def process_file(input_file)
    temp_output = nil
    
    begin
      # Check if file is valid PNG
      unless valid_png?(input_file)
        raise "File is not a valid PNG image"
      end
      
      # Check if image has transparency
      unless has_transparency?(input_file)
        raise "Image has no transparent pixels"
      end
      
      # Get histogram of existing colors
      puts "  → Analyzing existing colors..."
      existing_colors = get_existing_colors(input_file)
      puts "  → Found #{existing_colors.size} unique colors in image"
      
      # Get image dimensions and analyze transparent regions
      puts "  → Analyzing transparent regions..."
      transparent_regions = count_transparent_regions(input_file)
      puts "  → Found #{transparent_regions} disconnected transparent region(s)"
      
      # Generate unique colors for each region
      unique_colors = generate_unique_colors(existing_colors, transparent_regions)
      puts "  → Generated #{unique_colors.length} unique color(s): #{unique_colors.map { |c| format_color(c) }.join(', ')}"
      
      # Create temporary output filename
      temp_output = generate_temp_filename(input_file)
      
      # Process the image to temp file
      puts "  → Replacing transparent pixels with unique colors..."
      replace_transparent_pixels(input_file, temp_output, unique_colors)
      
      # Add alpha channel back
      puts "  → Restoring alpha channel..."
      add_alpha_channel(temp_output)
      
      # Verify temp output was created
      unless File.exist?(temp_output)
        raise "Temporary output file was not created"
      end
      
      # Replace original with processed version
      puts "  → Saving to original filename..."
      FileUtils.mv(temp_output, input_file)
      temp_output = nil # Clear since we've moved it
      
      puts "  ✓ SUCCESS: Processed #{File.basename(input_file)}"
      @stats[:files_processed] += 1
      
    rescue => e
      puts "  ✗ ERROR: #{e.message}"
      @stats[:files_errored] += 1
      @stats[:errors] << { file: File.basename(input_file), error: e.message }
      
      # Clean up temp file if it exists
      File.delete(temp_output) if temp_output && File.exist?(temp_output)
    end
  end
  
  def exec_magick(command)
    # Execute ImageMagick command and return output
    output = `#{command}`
    output
  end
  
  def valid_png?(file)
    result = exec_magick("magick identify -format \"%m\" \"#{file}\"").strip
    result == "PNG"
  end
  
  def has_transparency?(file)
    # Check image type - look for "Alpha" in the type
    result = exec_magick("magick identify -format \"%[type]\" \"#{file}\"").strip
    puts "  [DEBUG] Image type: '#{result}'" if ENV['DEBUG']
    
    # Check if it's a type with alpha channel
    return false unless result.include?("Alpha")
    
    # Check alpha channel min value - if min is less than max, there's transparency
    alpha_min = exec_magick("magick \"#{file}\" -channel A -separate -format \"%[min]\" info:").strip
    alpha_max = exec_magick("magick \"#{file}\" -channel A -separate -format \"%[max]\" info:").strip
    
    puts "  [DEBUG] Alpha min: '#{alpha_min}', max: '#{alpha_max}'" if ENV['DEBUG']
    
    return false if alpha_min.empty? || alpha_max.empty?
    
    # If min is less than max, we have varying transparency
    alpha_min.to_f < alpha_max.to_f
  end
  
  def get_existing_colors(file)
    # Get histogram: count for each unique RGB color (ignoring alpha)
    histogram_output = exec_magick("magick \"#{file}\" -alpha off -format %c histogram:info:-")
    
    colors = Set.new
    
    # Parse histogram output
    # Format example: "1234: (255,128,0) #FF8000 srgb(255,128,0)"
    histogram_output.each_line do |line|
      if line =~ /\((\d+),(\d+),(\d+)\)/
        r, g, b = $1.to_i, $2.to_i, $3.to_i
        colors.add([r, g, b])
      end
    end
    
    colors
  end
  
  def count_transparent_regions(file)
    # Create a temporary mask of transparent areas
    temp_mask = "temp_mask_#{Process.pid}.png"
    
    begin
      # Extract alpha channel, invert it (so transparent areas become white)
      # and create binary mask
      exec_magick("magick \"#{file}\" -channel A -separate -negate -threshold 50% \"#{temp_mask}\"")
      
      # Use connected components labeling to count regions
      result = exec_magick("magick \"#{temp_mask}\" -define connected-components:verbose=true -define connected-components:area-threshold=1 -connected-components 4 null:")
      
      puts "  [DEBUG] Connected components output:\n#{result}" if ENV['DEBUG']
      
      # Count objects (excluding background)
      # Format includes lines like "  0: 1280x748+0+0 639.5,373.5 123456 srgb(0,0,0)"
      count = 0
      result.each_line do |line|
        # Match lines that start with spaces and a number followed by colon
        if line =~ /^\s+(\d+):/
          object_id = $1.to_i
          count += 1 if object_id > 0  # Don't count background (id=0)
        end
      end
      
      # Ensure at least 1 region if we found transparent pixels
      [count, 1].max
      
    ensure
      File.delete(temp_mask) if File.exist?(temp_mask)
    end
  end
  
  def generate_unique_colors(existing_colors, count)
    unique_colors = []
    generator = COLOR_SCHEMES[@color_scheme]
    
    # Try to generate colors using the scheme
    max_attempts = (DARK_COLOR_MAX + 1) ** 3
    attempt = 0
    
    while unique_colors.length < count && attempt < max_attempts
      color = generator.call(attempt)
      
      # Clamp color values to valid range
      color = color.map { |c| [[c, 0].max, 255].min }
      
      # Skip if this color exists in the image
      unless existing_colors.include?(color)
        unique_colors << color
      end
      
      attempt += 1
    end
    
    # If we couldn't find enough with the scheme, fall back to systematic search
    if unique_colors.length < count
      (0..DARK_COLOR_MAX).each do |r|
        (0..DARK_COLOR_MAX).each do |g|
          (0..DARK_COLOR_MAX).each do |b|
            next if existing_colors.include?([r, g, b])
            next if unique_colors.include?([r, g, b])
            
            unique_colors << [r, g, b]
            return unique_colors if unique_colors.length >= count
          end
        end
      end
    end
    
    # If still not enough, expand beyond dark range
    if unique_colors.length < count
      (DARK_COLOR_MAX + 1..255).each do |r|
        (0..255).each do |g|
          (0..255).each do |b|
            next if existing_colors.include?([r, g, b])
            next if unique_colors.include?([r, g, b])
            
            unique_colors << [r, g, b]
            return unique_colors if unique_colors.length >= count
          end
        end
      end
    end
    
    unique_colors
  end
  
  def format_color(rgb)
    "RGB(#{rgb[0]},#{rgb[1]},#{rgb[2]})"
  end
  
  # MODIFIED: Generate temporary filename instead of output filename
  def generate_temp_filename(input_file)
    dir = File.dirname(input_file)
    basename = File.basename(input_file, ".png")
    # Use process ID to make it unique
    File.join(dir, "#{basename}_temp_#{Process.pid}.png")
  end
  
  def replace_transparent_pixels(input_file, output_file, unique_colors)
    # Use the first unique color as the base background
    r, g, b = unique_colors[0]
    
    puts "  [DEBUG] Using primary fill color: RGB(#{r},#{g},#{b})" if ENV['DEBUG']
    
    # Fill transparent pixels with the primary color
    exec_magick("magick \"#{input_file}\" -background \"rgb(#{r},#{g},#{b})\" -alpha remove \"#{output_file}\"")
  end
  
  def add_alpha_channel(file)
    # Add alpha channel back to the image
    # Use -alpha set to ensure alpha channel exists
    temp_output = "temp_alpha_#{Process.pid}.png"
    
    begin
      exec_magick("magick \"#{file}\" -alpha set \"#{temp_output}\"")
      
      # Replace original with alpha-enabled version
      if File.exist?(temp_output)
        FileUtils.mv(temp_output, file)
      end
    ensure
      File.delete(temp_output) if File.exist?(temp_output)
    end
  end
  
  def print_summary
    puts "=" * 70
    puts "PROCESSING SUMMARY"
    puts "=" * 70
    puts "Files successfully processed: #{@stats[:files_processed]}"
    puts "Files skipped: #{@stats[:files_skipped]}"
    puts "Files with errors: #{@stats[:files_errored]}"
    puts
    
    if @stats[:errors].any?
      puts "ERRORS:"
      @stats[:errors].each do |error|
        puts "  • #{error[:file]}: #{error[:error]}"
      end
      puts
    end
    
    if @stats[:files_processed] > 0
      if @fixup_mode
        puts "✓ Fixup complete! Files have been updated with magenta backgrounds."
      else
        puts "✓ Processing complete! Files have been updated in-place."
      end
    end
    puts "=" * 70
  end
  
  def command_exists?(command)
    `where #{command} >nul 2>nul`
    $?.success?
  end
end

# Parse command line options
options = {}
color_scheme = :dark  # Default
fixup_mode = false

OptionParser.new do |opts|
  opts.banner = "Usage: ruby spritesheet_prep.rb [options]"
  opts.separator ""
  opts.separator "Process PNG spritesheets to replace transparency with unique dark colors"
  opts.separator "NOTE: This will overwrite the original files!"
  opts.separator ""
  opts.separator "Required (mutually exclusive):"
  
  opts.on("-i", "--image FILE", "Process a single image file") do |file|
    options[:image] = file
  end
  
  opts.on("-d", "--dir DIRECTORY", "Process all PNG files in directory") do |dir|
    options[:dir] = dir
  end
  
  opts.separator ""
  opts.separator "Modes:"
  
  opts.on("--fixup", "Fixup mode: Replace transparency with MAGENTA in *_45deg.png files") do
    fixup_mode = true
  end
  
  opts.separator ""
  opts.separator "Color Scheme (mutually exclusive, optional, ignored in fixup mode):"
  
  opts.on("--magenta", "Use magenta-biased dark colors (higher R+B, lower G)") do
    color_scheme = :magenta
  end
  
  opts.on("--cyan", "Use cyan-biased dark colors (higher G+B, lower R)") do
    color_scheme = :cyan
  end
  
  opts.on("--gray", "Use grayscale dark colors (equal RGB values)") do
    color_scheme = :gray
  end
  
  opts.separator ""
  opts.separator "Other:"
  
  opts.on("-h", "--help", "Show this help message") do
    puts opts
    puts ""
    puts "Examples:"
    puts "  # Normal mode - process all PNGs with dark colors"
    puts "  ruby spritesheet_prep.rb --dir E:\\sprite-data\\raw"
    puts ""
    puts "  # Fixup mode - fix all *_45deg.png files with magenta"
    puts "  ruby spritesheet_prep.rb --fixup --dir E:\\sprite-data\\raw"
    puts ""
    puts "  # Fixup single file"
    puts "  ruby spritesheet_prep.rb --fixup --image E:\\sprite-data\\raw\\sprite_45deg.png"
    exit
  end
end.parse!

# Validate mutually exclusive input options
if options[:image] && options[:dir]
  abort "ERROR: --image and --dir are mutually exclusive. Use only one."
end

unless options[:image] || options[:dir]
  abort "ERROR: Must specify either --image or --dir.\nUse --help for usage information."
end

# Validate mutually exclusive color scheme options (ignored in fixup mode)
unless fixup_mode
  color_flags = [options[:magenta], options[:cyan], options[:gray]].compact
  if color_flags.length > 1
    abort "ERROR: Color scheme options (--magenta, --cyan, --gray) are mutually exclusive. Use only one."
  end
end

# Run the processor
prepper = SpritesheetPrepper.new(color_scheme, fixup_mode)
prepper.run(options)
