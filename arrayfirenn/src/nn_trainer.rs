use self::HaltCondition::{ Epochs, MSE, Timer };
use self::LearningMode::{ Incremental };

use std::slice;
use time::{ Duration, PreciseTime };

use super::nn::NN;

pub static DEFAULT_LEARNING_RATE: f32 = 0.3f32;
pub static DEFAULT_MOMENTUM: f32 = 0f32;
pub static DEFAULT_EPOCHS: u32 = 1000;

/// Specifies when to stop training the network
#[derive(Debug, Copy, Clone)]
pub enum HaltCondition {
    /// Stop training after a certain number of epochs
    Epochs(u32),
    /// Train until a certain error rate is achieved
    MSE(f32),
    /// Train for some fixed amount of time and then halt
    Timer(Duration),
}

/// Specifies which [learning mode](http://en.wikipedia.org/wiki/Backpropagation#Modes_of_learning) to use when training the network
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum LearningMode {
    /// train the network Incrementally (updates weights after each example)
    Incremental
}

/// Used to specify options that dictate how a network will be trained
pub struct Trainer<'a,'b> {
    examples: &'b [(Vec<f32>, Vec<f32>)],
    rate: f32,
    momentum: f32,
    log_interval: Option<u32>,
    halt_condition: HaltCondition,
    learning_mode: LearningMode,
    nn: &'a mut NN,
}

/// `Trainer` is used to chain together options that specify how to train a network.
/// All of the options are optional because the `Trainer` struct
/// has default values built in for each option. The `go()` method must
/// be called however or the network will not be trained.
impl<'a,'b> Trainer<'a,'b>  {

    /// Specifies the learning rate to be used when training (default is `0.3`)
    /// This is the step size that is used in the backpropagation algorithm.
    pub fn rate(&mut self, rate: f32) -> &mut Trainer<'a,'b> {
        if rate <= 0f32 {
            panic!("the learning rate must be a positive number");
        }

        self.rate = rate;
        self
    }

    /// Specifies the momentum to be used when training (default is `0.0`)
    pub fn momentum(&mut self, momentum: f32) -> &mut Trainer<'a,'b> {
        if momentum <= 0f32 {
            panic!("momentum must be positive");
        }

        self.momentum = momentum;
        self
    }

    /// Specifies how often (measured in batches) to log the current error rate (mean squared error) during training.
    /// `Some(x)` means log after every `x` batches and `None` means never log
    pub fn log_interval(&mut self, log_interval: Option<u32>) -> &mut Trainer<'a,'b> {
        match log_interval {
            Some(interval) if interval < 1 => {
                panic!("log interval must be Some positive number or None")
            }
            _ => ()
        }

        self.log_interval = log_interval;
        self
    }

    /// Specifies when to stop training. `Epochs(x)` will stop the training after
    /// `x` epochs (one epoch is one loop through all of the training examples)
    /// while `MSE(e)` will stop the training when the error rate
    /// is at or below `e`. `Timer(d)` will halt after the [duration](https://doc.rust-lang.org/time/time/struct.Duration.html) `d` has
    /// elapsed.
    pub fn halt_condition(&mut self, halt_condition: HaltCondition) -> &mut Trainer<'a,'b> {
        match halt_condition {
            Epochs(epochs) if epochs < 1 => {
                panic!("must train for at least one epoch")
            }
            MSE(mse) if mse <= 0f32 => {
                panic!("MSE must be greater than 0")
            }
            _ => ()
        }

        self.halt_condition = halt_condition;
        self
    }
    /// Specifies what [mode](http://en.wikipedia.org/wiki/Backpropagation#Modes_of_learning) to train the network in.
    /// `Incremental` means update the weights in the network after every example.
    pub fn learning_mode(&mut self, learning_mode: LearningMode) -> &mut Trainer<'a,'b> {
        self.learning_mode = learning_mode;
        self
    }

    /// When `go` is called, the network will begin training based on the
    /// options specified. If `go` does not get called, the network will not
    /// get trained!
    pub fn go(&mut self) -> f32 {
        self.nn.train_details(
            self.examples,
            self.rate,
            self.momentum,
            self.log_interval,
            self.halt_condition
        )
    }

}
