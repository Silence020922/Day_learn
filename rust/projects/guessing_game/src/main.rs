use rand::Rng;
use std::cmp::Ordering;
use std::io;

fn main() {
    println!("Guess the number!");
    let chance_num = 6; // 尝试次数
    let mut times = 0;
    let target_number = rand::thread_rng().gen_range(1..=100); //注意包含上下端点
    loop {
        times = times + 1;
        println!("times:{}/chance:{}. Input your guess",times,chance_num);
        match times.cmp(&chance_num) {
            Ordering::Greater => {
                println!("You loss!");
                break;
            },
            _ => (),
        }
        let mut guess = String::new(); // let创建变量 mut标志变量可变 new创建类型实例
        io::stdin() //调用io库包中的stdin()
            .read_line(&mut guess) //read_line获取输入句柄并追加到参数变量中同时返回错误信息
            .expect("Failed to read line"); //expect处理错误信息
        let guess: u32 = match guess.trim().parse() {
            Ok(num) => num,
            Err(_) => continue,
        }; // 转换输入类型 trim去除字符串开头和结尾的空白字符,由于外部输入5回车后会得到5\n parse标志转换数据类型 u32无符号32整形 guess:指定类型
        println!("You guessed: {guess}");
        // match 分支 cmp作比较并返回Ordering中的成员
        match guess.cmp(&target_number) {
            //Ordering 是一个枚举，他的成员是Less,Greater,Equal
            Ordering::Less => println!("Too small!"),
            Ordering::Greater => println!("Too big!"),
            Ordering::Equal => {
                println!("You win!");
                break;
            }
        }
    }
}
