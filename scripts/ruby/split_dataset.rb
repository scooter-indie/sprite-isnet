#!/usr/bin/env ruby
# split_dataset.rb - Split data into train/valid/test sets

require 'fileutils'

class DatasetSplitter
  def initialize(source_dir, data_root, train_ratio: 0.7, valid_ratio: 0.15, test_ratio: 0.15, seed: 42)
    @source_dir = source_dir
    @data_root = data_root
    @train_ratio = train_ratio
    @valid_ratio = valid_ratio
    @test_ratio = test_ratio
    @seed = seed
    
    # Validate ratios
    total = train_ratio + valid_ratio + test_ratio
    unless (total - 1.0).abs < 0.001
      raise "Ratios must sum to 1.0 (got #{total})"
    end
  end
  
  def split
    puts "╔═══════════════════════════════════════════════════════════════╗"
    puts "║         Dataset Splitter                                      ║"
    puts "╚═══════════════════════════════════════════════════════════════╝"
    puts ""
    
    # Get all images from source
    images = Dir.glob(File.join(@source_dir, 'images', '*.png')).sort
    masks = Dir.glob(File.join(@source_dir, 'masks', '*.png')).sort
    
    puts "Found #{images.length} images and #{masks.length} masks"
    
    # Verify pairs exist
    valid_pairs = []
    images.each do |img_path|
      basename = File.basename(img_path, '.png')
      mask_path = File.join(@source_dir, 'masks', "#{basename}.png")
      
      if File.exist?(mask_path)
        valid_pairs << { image: img_path, mask: mask_path, basename: basename }
      else
        puts "⚠ Warning: No mask for #{basename}"
      end
    end
    
    puts "Valid pairs: #{valid_pairs.length}"
    
    if valid_pairs.empty?
      puts "✗ No valid image/mask pairs found"
      return
    end
    
    # Shuffle with seed
    srand(@seed)
    valid_pairs.shuffle!
    
    # Calculate split indices
    total = valid_pairs.length
    train_count = (total * @train_ratio).round
    valid_count = (total * @valid_ratio).round
    test_count = total - train_count - valid_count
    
    puts ""
    puts "Split:"
    puts "  Training:   #{train_count} (#{(@train_ratio * 100).round(1)}%)"
    puts "  Validation: #{valid_count} (#{(@valid_ratio * 100).round(1)}%)"
    puts "  Test:       #{test_count} (#{(@test_ratio * 100).round(1)}%)"
    puts ""
    
    # Split data
    train_pairs = valid_pairs[0...train_count]
    valid_pairs_data = valid_pairs[train_count...(train_count + valid_count)]
    test_pairs = valid_pairs[(train_count + valid_count)..-1]
    
    # Copy to directories
    copy_pairs(train_pairs, 'train')
    copy_pairs(valid_pairs_data, 'valid')
    copy_pairs(test_pairs, 'test')
    
    puts ""
    puts "✓ Dataset split complete!"
    puts ""
    puts "Output structure:"
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
  end
  
  def copy_pairs(pairs, subset_name)
    images_dir = File.join(@data_root, subset_name, 'images')
    masks_dir = File.join(@data_root, subset_name, 'masks')
    
    FileUtils.mkdir_p(images_dir)
    FileUtils.mkdir_p(masks_dir)
    
    print "Copying #{subset_name} set (#{pairs.length} pairs)... "
    
    pairs.each do |pair|
      dest_image = File.join(images_dir, "#{pair[:basename]}.png")
      dest_mask = File.join(masks_dir, "#{pair[:basename]}.png")
      
      FileUtils.cp(pair[:image], dest_image)
      FileUtils.cp(pair[:mask], dest_mask)
    end
    
    puts "✓"
  end
end

# CLI
if __FILE__ == $0
  require 'optparse'
  
  options = {
    train: 0.7,
    valid: 0.15,
    test: 0.15,
    seed: 42
  }
  
  OptionParser.new do |opts|
    opts.banner = "Usage: ruby split_dataset.rb [options] <source_dir> <data_root>"
    
    opts.on("--train RATIO", Float, "Training ratio (default: 0.7)") do |v|
      options[:train] = v
    end
    
    opts.on("--valid RATIO", Float, "Validation ratio (default: 0.15)") do |v|
      options[:valid] = v
    end
    
    opts.on("--test RATIO", Float, "Test ratio (default: 0.15)") do |v|
      options[:test] = v
    end
    
    opts.on("--seed SEED", Integer, "Random seed (default: 42)") do |v|
      options[:seed] = v
    end
    
    opts.on("-h", "--help", "Show this help") do
      puts opts
      exit
    end
  end.parse!
  
  if ARGV.length < 2
    puts "Error: Missing required arguments"
    puts ""
    puts "Usage: ruby split_dataset.rb [options] <source_dir> <data_root>"
    puts ""
    puts "Example:"
    puts "  ruby split_dataset.rb C:\\sprite-data\\processed C:\\sprite-data"
    puts ""
    puts "This will create train/valid/test splits in data_root"
    exit 1
  end
  
  source_dir = ARGV[0]
  data_root = ARGV[1]
  
  splitter = DatasetSplitter.new(
    source_dir,
    data_root,
    train_ratio: options[:train],
    valid_ratio: options[:valid],
    test_ratio: options[:test],
    seed: options[:seed]
  )
  
  splitter.split
end
