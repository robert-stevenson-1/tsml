package tsml.classifiers.distance_based.utils.system.timing;

import java.util.concurrent.TimeUnit;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import static utilities.Utilities.sleep;

public class StopWatchTest {

    private StopWatch stopWatch;

    @Before public void before() {
        stopWatch = new StopWatch();
    }

    @Test
    public void testGetPreviousElapsedTime() {
        final long time = System.nanoTime();
        stopWatch.start();
        sleep(100);
        // lap / split not called yet so elapsed time should be 0
        Assert.assertEquals(time, stopWatch.getElapsedTime(), 0);
        stopWatch.stop();
        Assert.assertEquals(time, stopWatch.getElapsedTime(), 1000);
    }
    
    @Test(expected = IllegalStateException.class)
    public void testGetStartTimeWhenStopped() {
        stopWatch.stop(false);
        Assert.assertFalse(stopWatch.isStarted());
        stopWatch.getPreviousElapsedTime();
    }

    @Test
    public void testGetStartTimeWhenStarted() {
        long timeStamp = System.nanoTime();
        stopWatch.start();
        Assert.assertTrue(stopWatch.isStarted());
        long startTime = stopWatch.getPreviousElapsedTime();
        Assert.assertTrue(startTime > timeStamp);
        Assert.assertTrue(startTime < timeStamp + TimeUnit.NANOSECONDS.convert(10, TimeUnit.MILLISECONDS));
    }

    @Test
    public void testReset() {
        stopWatch.start();
        stopWatch.stop();
        stopWatch.reset();
        Assert.assertEquals(stopWatch.getElapsedTimeStopped(), 0);
        Assert.assertEquals(stopWatch.getSplitTimeStopped(), 0);
    }

    @Test
    public void testResetTime() {
        stopWatch.start();
        stopWatch.resetElapsedTime();
        Assert.assertNotEquals(stopWatch.lap(), 0);
        stopWatch.stop();
        Assert.assertNotEquals(stopWatch.getElapsedTimeStopped(), 0);
    }

    @Test(expected = IllegalStateException.class)
    public void testGetElapsedTimeNotStopped() {
        stopWatch.start();
        stopWatch.resetElapsedTime();
        Assert.assertNotEquals(stopWatch.lap(), 0);
        Assert.assertNotEquals(stopWatch.getElapsedTimeStopped(), 0);
    }

    @Test
    public void testResetClock() {
        stopWatch.start();
        long startTime = stopWatch.lap();
        stopWatch.resetElapsedTime();
        Assert.assertTrue(stopWatch.getPreviousElapsedTime() > startTime);
    }

    @Test
    public void testLap() throws InterruptedException {
        long sleepTime = TimeUnit.NANOSECONDS.convert(100, TimeUnit.MILLISECONDS);
        long tolerance = TimeUnit.NANOSECONDS.convert(500, TimeUnit.MILLISECONDS);
//        System.out.println("t: " + tolerance);
//        System.out.println("s: " + sleepTime);
        stopWatch.start();
        for(int i = 1; i <= 5; i++) {
            long sleep = TimeUnit.MILLISECONDS.convert(sleepTime, TimeUnit.NANOSECONDS);
            Thread.sleep(sleep);
            long lapTime = stopWatch.lap();
//            System.out.println("l: " + lapTime);
            Assert.assertTrue(lapTime > sleepTime * i );
            Assert.assertTrue(lapTime < (sleepTime + tolerance) * i);
        }
    }

    @Test
    public void testStop() {
        stopWatch.start();
        long startTime = stopWatch.getPreviousElapsedTime();
        Assert.assertTrue(stopWatch.isStarted());
        stopWatch.stop();
        long stopTime = stopWatch.getElapsedTimeStopped();
        Assert.assertTrue(stopTime > 0);
        Assert.assertFalse(stopWatch.isStarted());
    }

    @Test
    public void testDoubleStop() {
        stopWatch.start();
        Assert.assertTrue(stopWatch.isStarted());
        stopWatch.stop();
        Assert.assertTrue(stopWatch.isStopped());
        stopWatch.stop(false);
        Assert.assertTrue(stopWatch.isStopped());
        try {
            stopWatch.stop();
            Assert.fail();
        } catch(IllegalStateException e) {

        }
        Assert.assertTrue(stopWatch.isStopped());
    }

    @Test
    public void testDoubleStart() {
        stopWatch.start();
        Assert.assertTrue(stopWatch.isStarted());
        stopWatch.start(false);
        Assert.assertTrue(stopWatch.isStarted());
        try {
            stopWatch.start();
            Assert.fail();
        } catch(IllegalStateException e) {

        }
        Assert.assertTrue(stopWatch.isStarted());
    }

    @Test
    public void testAdd() {
        stopWatch.start();
        stopWatch.stop();
        long time = stopWatch.getElapsedTimeStopped();
        long addend = 10;
        stopWatch.add(addend);
        Assert.assertEquals(addend + time, stopWatch.getElapsedTimeStopped());
        long prevTime = stopWatch.getElapsedTimeStopped();
        stopWatch.add(stopWatch);
        Assert.assertEquals(prevTime * 2, stopWatch.getElapsedTimeStopped());
    }

    @Test
    public void testSplit() {
        stopWatch.start();
        try {
            Thread.sleep(100);
        } catch(InterruptedException e) {
            e.printStackTrace();
        }
        long split1 = stopWatch.split();
        Assert.assertEquals(stopWatch.getElapsedTime(), split1);
        try {
            Thread.sleep(100);
        } catch(InterruptedException e) {
            e.printStackTrace();
        }
        long split2 = stopWatch.split();
        Assert.assertEquals(stopWatch.getElapsedTime(), split1 + split2);
        try {
            Thread.sleep(100);
        } catch(InterruptedException e) {
            e.printStackTrace();
        }
        long split3 = stopWatch.split();
        Assert.assertEquals(stopWatch.getElapsedTime(), split1 + split2 + split3);
    }

    @Test(expected = IllegalStateException.class)
    public void testLapWhenStopped() {
        stopWatch.stop();
        stopWatch.lap();
    }
}
