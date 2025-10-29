require 'fileutils'

data_root = ARGV[0] || 'E:\Projects\sprite-data'

puts "Data root: #{data_root}"
puts "Data root exists?: #{Dir.exist?(data_root)}"
puts ""

source_images = File.join(data_root, 'train', 'images')
puts "Source images path: #{source_images}"
puts "Source images exists?: #{Dir.exist?(source_images)}"
puts ""

# Try different approaches
puts "Approach 1: Dir.glob with File.join"
files1 = Dir.glob(File.join(source_images, '*'))
puts "  Found: #{files1.length} total files"
puts "  First 3: #{files1.first(3)}"
puts ""

puts "Approach 2: Dir.glob with string interpolation"
files2 = Dir.glob("#{source_images}/*")
puts "  Found: #{files2.length} total files"
puts "  First 3: #{files2.first(3)}"
puts ""

puts "Approach 3: Dir.glob with normalized path"
normalized = source_images.gsub('\\', '/')
files3 = Dir.glob("#{normalized}/*")
puts "  Normalized path: #{normalized}"
puts "  Found: #{files3.length} total files"
puts "  First 3: #{files3.first(3)}"
puts ""

puts "Approach 4: Dir.entries"
if Dir.exist?(source_images)
  entries = Dir.entries(source_images).reject { |f| f == '.' || f == '..' }
  puts "  Found: #{entries.length} entries"
  puts "  First 3: #{entries.first(3)}"
end
