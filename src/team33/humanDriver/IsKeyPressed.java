package team33.humanDriver;

import java.awt.Frame;
import java.awt.KeyEventDispatcher;
import java.awt.KeyboardFocusManager;
import java.awt.event.KeyEvent;

public class IsKeyPressed {
	private static final double MAX_STEERING_TIME_MS = 1000;
	
    private boolean upPressed = false;
    public boolean isUpPressed() {
        synchronized (IsKeyPressed.class) {
            return upPressed;
        }
    }
    
    private long upPressedTime;
    public double upAmount() {
    	synchronized (IsKeyPressed.class) {
    		return (double) Math.min((System.currentTimeMillis() - upPressedTime) / MAX_STEERING_TIME_MS, 1.0);
    	}
    }
    
    private boolean downPressed = false;
    public boolean isDownPressed() {
        synchronized (IsKeyPressed.class) {
            return downPressed;
        }
    }
    
    private long downPressedTime;
    public double downAmount() {
    	synchronized (IsKeyPressed.class) {
    		return (double) Math.min((System.currentTimeMillis() - downPressedTime) / MAX_STEERING_TIME_MS, 1.0);
    	}
    }
    
    private boolean leftPressed = false;
    public boolean isLeftPressed() {
        synchronized (IsKeyPressed.class) {
            return leftPressed;
        }
    }
    
    private long leftPressedTime;
    public double leftAmount() {
    	synchronized (IsKeyPressed.class) {
    		return (double) Math.min((System.currentTimeMillis() - leftPressedTime) / MAX_STEERING_TIME_MS, 1.0);
    	}
    }
    
    private boolean rightPressed = false;
    public boolean isRightPressed() {
        synchronized (IsKeyPressed.class) {
            return rightPressed;
        }
    }
    
    private long rightPressedTime;
    public double rightAmount() {
    	synchronized (IsKeyPressed.class) {
    		return (double) Math.min((System.currentTimeMillis() - rightPressedTime) / MAX_STEERING_TIME_MS, 1.0);
    	}
    }

    public IsKeyPressed() {
    	Frame frame = new Frame("KeyEvent grabber");
    	frame.setSize(300, 200);
    	frame.setVisible(true);
        KeyboardFocusManager.getCurrentKeyboardFocusManager().addKeyEventDispatcher(new KeyEventDispatcher() {

            @Override
            public boolean dispatchKeyEvent(KeyEvent ke) {
                synchronized (IsKeyPressed.class) {
                    switch (ke.getID()) {
                    case KeyEvent.KEY_PRESSED:
                    	if (ke.getKeyCode() == KeyEvent.VK_UP) {
                            if (!upPressed)
                            	upPressedTime = System.currentTimeMillis();
                    		upPressed = true;
                        }
                    	if (ke.getKeyCode() == KeyEvent.VK_DOWN) {
                            if (!downPressed)
                            	downPressedTime = System.currentTimeMillis();
                    		downPressed = true;
                        }
                        if (ke.getKeyCode() == KeyEvent.VK_LEFT) {
							if (!leftPressed)
								leftPressedTime = System.currentTimeMillis();
                        	leftPressed = true;
						}
                        if (ke.getKeyCode() == KeyEvent.VK_RIGHT) {
							if (!rightPressed)
								rightPressedTime = System.currentTimeMillis();
                        	rightPressed = true;
						}
                        break;

                    case KeyEvent.KEY_RELEASED:
                        if (ke.getKeyCode() == KeyEvent.VK_UP) {
                            upPressed = false;
                        }
                        if (ke.getKeyCode() == KeyEvent.VK_DOWN) {
                            downPressed = false;
                        }
                        if (ke.getKeyCode() == KeyEvent.VK_LEFT) {
                        	leftPressed = false;
                        }
                        if (ke.getKeyCode() == KeyEvent.VK_RIGHT) {
                        	rightPressed = false;
                        }
                        break;
                    }
                    return false;
                }
            }
        });
    }
}