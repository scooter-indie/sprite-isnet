#!/usr/bin/env ruby
# split_train_data.rb - Split existing train data into train/valid/test sets
# Works with data already in train/ folder

require 'fileutils'

class TrainDataSplitter
  def initialize(data_root, train_ratio: 0.70, valid_ratio: 0.15, test_ratio: 0.15, seed: 42)
    @data_root = data_root
    @train_ratio = train_ratio
    @valid_ratio = valid_ratio
    @test_ratio = test_ratio
    @seed = seed
    
    # Source directories (where all data currently is)
    @source_images = File.join(@data_root, 'train', 'images')
    @source_masks = File.join(@data_root, 'train', 'masks')
    
    # Destination directories
    @train_images = File.join(@data_root, 'train', 'images')
    @train_masks = File.join(@data_root, 'train', 'masks')
    @valid_images = File.join(@data_root, 'valid', 'images')
    @valid_masks = File.join(@data_root, 'valid', 'masks')
    @test_images = File.join(@data_root, 'test', 'images')
    @test_masks = File.join(@data_root, 'test', 'masks')
    
    # Validate ratios
    total = train_ratio + valid_ratio + test_ratio
    unless (total - 1.0).abs < 0.001
      raise "Ratios must sum to 1.0 (got #{total})"
    end
  end
  
  def split
    puts "╔═══════════════════════════════════════════════════════════════╗"
    puts "║         Split Training Data into Train/Valid/Test            ║"
    puts "╚═══════════════════════════════════════════════════════════════╝"
    puts ""
    
    # Check source directories exist
    unless Dir.exist?(@source_images)
      puts "✗ Source images directory not found: #{@source_images}"
      exit 1
    end
    
    unless Dir.exist?(@source_masks)
      puts "✗ Source masks directory not found: #{@source_masks}"
      exit 1
    end
    
    # Get all images from train directory
    image_files = Dir.glob(File.join(@source_images, '*.{png,jpg,jpeg,PNG,JPG,JPEG}'), File::FNM_EXTGLOB).sort
    
    puts "Source: #{@source_images}"
    puts "Found #{image_files.length} images"
    puts ""
    
    if image_files.empty?
      puts "✗ No images found in source directory"
      exit 1
    end
    
    # Verify pairs exist
    valid_pairs = []
    missing_masks = []
    
    image_files.each do |img_path|
      basename = File.basename(img_path, File.extname(img_path))
      mask_path = File.join(@source_masks, "#{basename}.png")
      
      if File.exist?(mask_path)
        valid_pairs << { 
          image: img_path, 
          mask: mask_path, 
          basename: basename,
          image_ext: File.extname(img_path)
        }
      else
        missing_masks << basename
      end
    end
    
    if missing_masks.any?
      puts "⚠ Warning: #{missing_masks.length} images missing corresponding masks:"
      missing_masks.first(5).each { |name| puts "  - #{name}" }
      puts "  ... and #{missing_masks.length - 5} more" if missing_masks.length > 5
      puts ""
    end
    
    puts "Valid image/mask pairs: #{valid_pairs.length}"
    
    if valid_pairs.empty?
      puts "✗ No valid image/mask pairs found"
      exit 1
    end
    
    # Check if we have enough samples
    if valid_pairs.length < 10
      puts "⚠ Warning: Very small dataset (#{valid_pairs.length} samples)"
      puts "  Consider having at least 50-100 samples for training"
      print "Continue anyway? (y/n): "
      response = gets.chomp.downcase
      exit 0 unless response == 'y'
    end
    
    # Shuffle with seed for reproducibility
    srand(@seed)
    valid_pairs.shuffle!
    
    # Calculate split indices
    total = valid_pairs.length
    train_count = (total * @train_ratio).round
    valid_count = (total * @valid_ratio).round
    test_count = total - train_count - valid_count
    
    puts ""
    puts "Split ratios:"
    puts "  Training:   #{train_count} samples (#{(@train_ratio * 100).round(1)}%)"
    puts "  Validation: #{valid_count} samples (#{(@valid_ratio * 100).round(1)}%)"
    puts "  Test:       #{test_count} samples (#{(@test_ratio * 100).round(1)}%)"
    puts ""
    
    # Confirmation
    print "Proceed with split? This will move files! (y/n): "
    response = gets.chomp.downcase
    unless response == 'y'
      puts "Cancelled."
      exit 0
    end
    
    # Split data
    train_pairs = valid_pairs[0...train_count]
    valid_pairs_data = valid_pairs[train_count...(train_count + valid_count)]
    test_pairs = valid_pairs[(train_count + valid_count)..-1]
    
    # Create destination directories
    FileUtils.mkdir_p(@valid_images)
    FileUtils.mkdir_p(@valid_masks)
    FileUtils.mkdir_p(@test_images)
    FileUtils.mkdir_p(@test_masks)
    
    puts ""
    puts "Moving files..."
    
    # Train data stays in place (no action needed)
    puts "✓ Training set: #{train_count} pairs (staying in train/)"
    
    # Move validation data
    move_pairs(valid_pairs_data, @valid_images, @valid_masks, 'validation')
    
    # Move test data
    move_pairs(test_pairs, @test_images, @test_masks, 'test')
    
    puts ""
    puts "╔═══════════════════════════════════════════════════════════════╗"
    puts "║         Dataset Split Complete!                              ║"
    puts "╚═══════════════════════════════════════════════════════════════╝"
    puts ""
    puts "Final structure:"
    puts "#{@data_root}"
    puts "├── train/"
    puts "│   ├── images/ (#{train_count} files)"
    puts "│   └── masks/  (#{train_count} files)"
    puts "├── valid/"
    puts "│   ├── images/ (#{valid_count} files)"
    puts "│   └── masks/  (#{valid_count} files)"
    puts "└── test/"
    puts "    ├── images/ (#{test_count} files)"
    puts "    └── masks/  (#{test_count} files)"
    puts ""
  end
  
  def move_pairs(pairs, dest_images_dir, dest_masks_dir, subset_name)
    print "Moving #{subset_name} set (#{pairs.length} pairs)... "
    
    pairs.each do |pair|
      # Preserve original image extension
      dest_image = File.join(dest_images_dir, "#{pair[:basename]}#{pair[:image_ext]}")
      dest_mask = File.join(dest_masks_dir, "#{pair[:basename]}.png")
      
      # Move files
      FileUtils.mv(pair[:image], dest_image)
      FileUtils.mv(pair[:mask], dest_mask)
    end
    
    puts "✓"
  end
end

# CLI
if __FILE__ == $0
  require 'optparse'
  
  options = {
    train: 0.70,
    valid: 0.15,
    test: 0.15,
    seed: 42
  }
  
  OptionParser.new do |opts|
    opts.banner = "Usage: ruby split_train_data.rb [options] <data_root>"
    
    opts.on("--train RATIO", Float, "Training ratio (default: 0.70)") do |v|
      options[:train] = v
    end
    
    opts.on("--valid RATIO", Float, "Validation ratio (default: 0.15)") do |v|
      options[:valid] = v
    end
    
    opts.on("--test RATIO", Float, "Test ratio (default: 0.15)") do |v|
      options[:test] = v
    end
    
    opts.on("--seed SEED", Integer, "Random seed for shuffling (default: 42)") do |v|
      options[:seed] = v
    end
    
    opts.on("-h", "--help", "Show this help") do
      puts opts
      puts ""
      puts "Example:"
      puts "  ruby split_train_data.rb E:\\Projects\\sprite-data"
      puts ""
      puts "This will:"
      puts "  1. Read all files from train/images and train/masks"
      puts "  2. Split them into train (70%), valid (15%), test (15%)"
      puts "  3. Move validation files to valid/"
      puts "  4. Move test files to test/"
      puts "  5. Leave remaining files in train/"
      puts ""
      exit
    end
  end.parse!
  
  if ARGV.length < 1
    puts "Error: Missing required argument <data_root>"
    puts ""
    puts "Usage: ruby split_train_data.rb [options] <data_root>"
    puts "Run with -h for help"
    exit 1
  end
  
  data_root = ARGV[0]
  
  unless Dir.exist?(data_root)
    puts "✗ Data root directory not found: #{data_root}"
    exit 1
  end
  
  splitter = TrainDataSplitter.new(
    data_root,
    train_ratio: options[:train],
    valid_ratio: options[:valid],
    test_ratio: options[:test],
    seed: options[:seed]
  )
  
  splitter.split
end
