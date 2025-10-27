#!/usr/bin/env ruby
# check_data_quality.rb - Validate dataset quality

require 'json'

class DataQualityChecker
  def initialize(data_root)
    @data_root = data_root
    @issues = []
    @stats = {
      train: { images: 0, masks: 0, paired: 0 },
      valid: { images: 0, masks: 0, paired: 0 },
      test: { images: 0, masks: 0, paired: 0 }
    }
  end
  
  def check_all
    puts "╔═══════════════════════════════════════════════════════════════╗"
    puts "║         Data Quality Checker                                  ║"
    puts "╚═══════════════════════════════════════════════════════════════╝"
    puts ""
    
    # Check each subset
    %w[train valid test].each do |subset|
      check_subset(subset)
    end
    
    print_report
  end
  
  def check_subset(subset)
    puts "[#{subset.upcase}]"
    
    images_dir = File.join(@data_root, subset, 'images')
    masks_dir = File.join(@data_root, subset, 'masks')
    
    # Check directories exist
    unless Dir.exist?(images_dir)
      @issues << "Missing directory: #{images_dir}"
      puts "  ✗ Images directory not found"
      return
    end
    
    unless Dir.exist?(masks_dir)
      @issues << "Missing directory: #{masks_dir}"
      puts "  ✗ Masks directory not found"
      return
    end
    
    # Get files
    images = Dir.glob(File.join(images_dir, '*.png')).sort
    masks = Dir.glob(File.join(masks_dir, '*.png')).sort
    
    @stats[subset.to_sym][:images] = images.length
    @stats[subset.to_sym][:masks] = masks.length
    
    puts "  Images: #{images.length}"
    puts "  Masks:  #{masks.length}"
    
    # Check for pairs
    paired = 0
    unpaired_images = []
    
    images.each do |img_path|
      basename = File.basename(img_path, '.png')
      mask_path = File.join(masks_dir, "#{basename}.png")
      
      if File.exist?(mask_path)
        paired += 1
        
        # Check mask quality
        check_mask_quality(img_path, mask_path, subset, basename)
      else
        unpaired_images << basename
      end
    end
    
    @stats[subset.to_sym][:paired] = paired
    
    if unpaired_images.empty?
      puts "  ✓ All images have corresponding masks"
    else
      puts "  ✗ #{unpaired_images.length} images without masks"
      @issues << "#{subset}: #{unpaired_images.length} unpaired images"
    end
    
    puts ""
  end
  
  def check_mask_quality(image_path, mask_path, subset, basename)
    # Get mask coverage
    coverage = `magick "#{mask_path}" -format "%[fx:mean*100]" info:`.strip.to_f
    
    # Check for issues
    if coverage < 2
      @issues << "#{subset}/#{basename}: Very low coverage (#{coverage.round(1)}%) - likely all background"
    elsif coverage > 98
      @issues << "#{subset}/#{basename}: Very high coverage (#{coverage.round(1)}%) - likely inverted mask"
    end
    
    # Check dimensions match
    img_size = `magick identify -format "%wx%h" "#{image_path}"`.strip
    mask_size = `magick identify -format "%wx%h" "#{mask_path}"`.strip
    
    if img_size != mask_size
      @issues << "#{subset}/#{basename}: Size mismatch - image: #{img_size}, mask: #{mask_size}"
    end
  end
  
  def print_report
    puts "=" * 60
    puts "DATASET SUMMARY"
    puts "=" * 60
    
    total_images = 0
    total_paired = 0
    
    %w[train valid test].each do |subset|
      stats = @stats[subset.to_sym]
      total_images += stats[:images]
      total_paired += stats[:paired]
      
      puts ""
      puts "#{subset.capitalize}:"
      puts "  Images:       #{stats[:images]}"
      puts "  Masks:        #{stats[:masks]}"
      puts "  Paired:       #{stats[:paired]}"
      
      if stats[:images] > 0
        ratio = (stats[:paired].to_f / stats[:images] * 100).round(1)
        puts "  Completeness: #{ratio}%"
      end
    end
    
    puts ""
    puts "Total Images: #{total_images}"
    puts "Total Paired: #{total_paired}"
    
    # Recommendations
    puts ""
    puts "=" * 60
    puts "RECOMMENDATIONS"
    puts "=" * 60
    
    if total_paired < 50
      puts "⚠ Very small dataset (#{total_paired} samples)"
      puts "  Recommended minimum: 100 samples"
      puts "  Better results with: 500+ samples"
    elsif total_paired < 200
      puts "⚠ Small dataset (#{total_paired} samples)"
      puts "  This is workable but consider adding more data"
    else
      puts "✓ Good dataset size (#{total_paired} samples)"
    end
    
    # Train/valid split
    train_count = @stats[:train][:paired]
    valid_count = @stats[:valid][:paired]
    
    if valid_count > 0
      split_ratio = (valid_count.to_f / (train_count + valid_count) * 100).round(1)
      puts ""
      puts "Validation split: #{split_ratio}%"
      
      if split_ratio < 10 || split_ratio > 30
        puts "  ⚠ Recommended: 10-20% for validation"
      else
        puts "  ✓ Good validation split"
      end
    end
    
    # Issues
    if @issues.empty?
      puts ""
      puts "=" * 60
      puts "✓ NO ISSUES FOUND - DATASET IS READY FOR TRAINING!"
      puts "=" * 60
    else
      puts ""
      puts "=" * 60
      puts "⚠ ISSUES FOUND (#{@issues.length})"
      puts "=" * 60
      @issues.each do |issue|
        puts "  - #{issue}"
      end
      puts ""
      puts "Please fix these issues before training"
    end
  end
end

# CLI
if __FILE__ == $0
  if ARGV.length < 1
    puts "Usage: ruby check_data_quality.rb <data_root>"
    puts ""
    puts "Example:"
    puts "  ruby check_data_quality.rb C:\\sprite-data"
    exit 1
  end
  
  data_root = ARGV[0]
  
  checker = DataQualityChecker.new(data_root)
  checker.check_all
end
