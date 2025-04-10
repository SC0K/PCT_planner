// Parameters
floor_height = 40;
room_w = 140;
room_d =70;
wall_thick = 2;
slab_thickness = 2;
stair_width = 20;
stair_height = floor_height;
stair_length = 70;
steps = 15;

door_w = 20;
door_h = 30;

stair_offset = -30;

// Create a single room (open front/back walls)
module room_open() {
    difference() {
        cube([room_w, room_d, floor_height], center=false);
        // Remove front and back walls
        translate([wall_thick, 0, 0])
            cube([room_w - 2 * wall_thick, room_d, floor_height]);
    }
}

// Create two rooms side by side with a doorway in the middle wall
module room_pair_with_doorway() {
    difference() {
        union() {
            room_open();
            translate([room_w, 0, 0]) room_open();
        }

        // Cut doorway in shared wall
        translate([
            room_w - wall_thick / 2 - door_w / 2,
            room_d / 2 - door_w / 2,
            0
        ])
        cube([door_w, door_w, door_h]);
    }
}

// Floor slab, optionally cut for stairs
module slab(z_offset, cut_for_stairs=false) {
    difference() {
        cube([2 * room_w, room_d, slab_thickness]);

        if (cut_for_stairs) {
            translate([
                2 * room_w - stair_length + stair_offset,  // leave some landing space
                0,
                -1  // ensure full cut
            ])
            cube([stair_length, stair_width, slab_thickness + 2]);
        }
    }
}

// Staircase
module staircase(x, y, z) {
    for (i = [0:steps - 1]) {
        translate([x + i * stair_length / steps, y, z + i * stair_height / steps])
            cube([stair_length / steps, stair_width, stair_height / steps]);
    }
}

// Complete building
module building() {
    // Ground floor rooms
    room_pair_with_doorway();
    slab(0);  // ground slab

    // First floor rooms
    translate([0, 0, floor_height + slab_thickness])
        room_pair_with_doorway();

    // First floor slab with stair cutout
    translate([0, 0, floor_height])
        slab(0, cut_for_stairs=true);

    // Stairs from ground to first floor
    translate([stair_offset,0,0])
    staircase(2 * room_w - stair_length, 0, slab_thickness);
}

// Render the building
scale([0.1, 0.1, 0.1])
translate([-room_w, -room_d/2])
building();
