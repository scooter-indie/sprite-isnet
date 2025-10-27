#!/usr/bin/env ruby
# prepare_dataset.rb - Master script for complete data preparation

require 'fileutils'
require 'optparse'

class DatasetPreparation
  def initialize(raw_dir, data_root, options = {})
    @raw_dir = raw_dir
    @data_root = data_root
    @options = options
    
    # Set defaults
    @method = options[:method] || 'auto'
    @fuzz = options[:fuzz] || 10
    @preview = options[:preview] || false
    @train_ratio = options[:train_ratio] || 0.7
    @valid_ratio = options[:valid_ratio] || 0.15
    @test_ratio = options[:test_ratio] || 0.15
    
    # Working directory
    @work_dir = File.join(@data_root, 'processed')
  end
  
  def run
    puts "╔═══════════════════════════════════════════════════════════════╗"
    puts "║         Complete Dataset Preparation                          ║"
    puts "╚═══════════════════════════════════════════════════════════════╝"
    puts ""
    
    # Step 1: Generate masks
    puts "STEP 1: Generate Masks"
    puts "=" * 60
    generate_masks
    puts ""
    
    # Step 2: Split dataset
    puts "STEP 2: Split Dataset"
    puts "=" * 60
    split_dataset
    puts ""
    
    # Step 3: Quality check
    puts "STEP 3: Quality Check"
    puts "=" * 60
    check_quality
    puts ""
    
    puts "╔═══════════════════════════════════════════════════════════════╗"
    puts "║         Dataset Preparation Complete!                         ║"
    puts "╚═══════════════════════════════════════════════════════════════╝"
  end
  
  def generate_masks
    images_dir = File.join(@work_dir, 'images')
    masks_dir = File.join(@work_dir, 'masks')
    
    cmd = [
      'ruby',
      'advanced_mask_generator.rb',
      '-m', @method,
      '-f', @fuzz.to_s
    ]
    
    cmd += ['-p'] if @preview
    cmd += [@raw_dir, images_dir, masks_dir]
    
    system(*cmd)
  end
  
  def split_dataset
    cmd = [
      'ruby',
      'split_dataset.rb',
      '--train', @train_ratio.to_s,
      '--valid', @valid_ratio.to_s,
      '--test', @test_ratio.to_s,
      @work_dir,
      @data_root
    ]
    
    system(*cmd)
  end
  
  def check_quality
    system('ruby', 'check_data_quality.rb', @data_root)
  end
end

# CLI
if __FILE__ == $0
  options = {}
  
  OptionParser.new do |opts|
    opts.banner = "Usage: ruby prepare_dataset.rb [options] <raw_dir> <data_root>"
    
    opts.on("-m", "--method METHOD", "Mask method: auto, corner, edge (default: auto)") do |v|
      options[:method] = v
    end
    
    opts.on("-f", "--fuzz PERCENT", Integer, "Color fuzz tolerance (default: 10)") do |v|
      options[:fuzz] = v
    end
    
    opts.on("-p", "--preview", "Generate preview images") do
      options[:preview] = true
    end
    
    opts.on("--train RATIO", Float, "Training ratio (default: 0.7)") do |v|
      options[:train_ratio] = v
    end
    
    opts.on("--valid RATIO", Float, "Validation ratio (default: 0.15)") do |v|
      options[:valid_ratio] = v
    end
    
    opts.on("--test RATIO", Float, "Test ratio (default: 0.15)") do |v|
      options[:test_ratio] = v
    end
    
    opts.on("-h", "--help", "Show this help") do
      puts opts
      puts ""
      puts "Example:"
      puts "  ruby prepare_dataset.rb -m auto -f 15 -p C:\\sprite-data\\raw C:\\sprite-data"
      exit
    end
  end.parse!
  
  if ARGV.length < 2
    puts "Error: Missing required arguments"
    puts ""
    puts "Usage: ruby prepare_dataset.rb [options] <raw_dir> <data_root>"
    puts "Run with -h for help"
    exit 1
  end
  
  raw_dir = ARGV[0]
  data_root = ARGV[1]
  
  prep = DatasetPreparation.new(raw_dir, data_root, options)
  prep.run
end
