fn main() {
    let s = String::from("Hello");
    let s = change(s);
    println!("{s}");
}

fn change(mut some_string: String)->String{
    some_string.push_str(", world.");
    some_string
}
