use std::io;

fn main() {
    println!("Please decide n:");
    let mut n = String::new();
    io::stdin()
        .read_line(&mut n)
        .expect("Failed to read line");
    let n: u32 = n.trim().parse().expect("n must be a number!");
    let mut i = 0;
    let mut x = 1;
    let mut y = 1;
    let mut element = 1;
    while i < n - 2 {
        i = i + 1;
        element = x + y;
        x = y;
        y = element;
    }
    println!("The value is {element}")
}
