module Main where

import Debug.Trace (trace, traceShowId)
import System.Environment (getArgs)

import Data.List (group, transpose)

data Value = Set | Unset | Unknow deriving (Eq, Show)
type Line = [Value]
type Constraint = [Int]

showValue :: Value -> Char
showValue Set = '■'
showValue Unset = 'x'
showValue Unknow = '_'

genAllPoseLoop :: Int -> Int -> [[Int]]
genAllPoseLoop nLeft 1 = [[nLeft]]
genAllPoseLoop nLeft pLeft = concat [map (this:) $ genAllPoseLoop (nLeft-this) (pLeft-1) | this <- [0..nLeft]]

genAllPose :: Constraint -> Int -> [[(Int, Int)]]
genAllPose c n = map (zip (0:c) . zipWith (+) (0 : replicate (lc-1) 1 ++ [0])) $ genAllPoseLoop (n - sum c - lc + 1) (lc + 1)
  where lc = length c

checkUnique :: Constraint -> Line -> Line
checkUnique con line = map checkSame $ transpose findAllCase
  where checkSame s = if all (== head s) s then head s else Unknow
        findAllCase = filter checkLine $ map applyPose $ genAllPose con (length line)
        checkLine l = and . zipWith (\a b -> (a == Unknow) || (a == b)) line $ l
        applyPose p = concat [replicate i Set ++ replicate j Unset | (i, j) <- p]

main :: IO ()
main = do
  putStrLn $ map showValue $ checkUnique [5] $ replicate 8 Unknow
