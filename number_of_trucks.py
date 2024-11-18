### Truck Capacity in 2d ###

from rectpack import newPacker
import pandas as pd

file_path = 'items_length_width.xlsx'
df = pd.read_excel(file_path)

# Truck dimensions
truck_width = 5
truck_length = 15


# Initialize packer with the truck dimensions
packer = newPacker()

# Add truck as a "bin"
packer.add_bin(truck_width, truck_length)

# Add all items as rectangles (with both orientations)
for i in range(len(df)):
    width = df.loc[i, 'width']
    length = df.loc[i, 'length']
    packer.add_rect(width, length)
    packer.add_rect(length, width)


packer.pack()

# Count the number of trucks used
print("Number of trucks needed is", len(packer))

# Show packing result
for b in packer:
    print("Truck:", b)
    for r in b:
        print("  Item:", r.width, "x", r.height, "at position", r.x, ",", r.y)