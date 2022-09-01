module Main where

import Debug.Trace (trace, traceShowId)
import System.Environment (getArgs)

import Data.List (group, transpose)
import GHC.Float (int2Double)

data Value = Set | Unset | Unknow deriving (Eq, Show)
type Line = [Value]
type Board = [Line]
type Constraint = [Int]
type Update = (Int, Value)

showValue :: Value -> Char
showValue Set = '■'
showValue Unset = 'x'
showValue Unknow = '_'

showBoard :: Board -> String
showBoard b = unlines $ map (map showValue) b

horizonConst :: [Constraint]
horizonConst = [[2,6],[1,1,6],[11],[1,3,2],[3,7,2],[1,1,5,1],[10],[7],[2,2],[1,2],[1,1,3],[2,3,2],[6,2,1],[2,1,5],[2,1,4]]

verticalConst :: [Constraint]
verticalConst = [[1,2,1,1,3],[1,1,3],[1,3,1],[3,2,4],[1,5,2],[1,1,4,1],[6],[6,1,2],[1,7,4],[3,11],[3,2,1,2],[3,1],[3,3],[5,2],[1,3,1]]

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

updateBoard :: [Bool] -> [Constraint] -> Board -> (Bool, [Bool], Board)
updateBoard ss cc ll = (any null res, map and . transpose $ zipWith (zipWith (==)) res ll, res)
  where res = [if s then l else checkUnique c l | (s, c, l) <- zip3 ss cc ll]

checkLoop :: (Int, [Bool], Board) -> (Int, Bool, [Bool], Board)
checkLoop (n, ss, board)
  | n <= 0 || and ss' || fail = (n-1, fail, ss', board')
  | otherwise = checkLoop (n-1, ss', board')
  where (fail, ss', b') = if isH then updateBoard ss horizonConst board
                          else updateBoard ss verticalConst (transpose board)
        board' = if isH then b' else transpose b'
        isH = even n

modifyAt :: [a] -> Int -> (a -> a) -> [a]
modifyAt xs i f = take i xs ++ [f $ xs !! i] ++ drop (i + 1) xs

randomChange :: Double -> Board -> (Double, Board)
randomChange rnd board = (rndNext nrnd, modifyAt board (rc rnd board) (\x -> modifyAt x (rc nrnd x) (const Set)))
  where rndNext r = if r >= 0.3 then (1-r)/0.7 else r/0.3
        nrnd = rndNext rnd
        rc r s = floor (r * int2Double (length s))

solve :: Double -> Int -> Board -> Board -> (Int, Bool, Bool, Board)
solve rnd n b ori
  | n' <= 0 || finish = (n', fail, not fail && finish, res)
  | otherwise = solve rnd' n' bGuess ori
  where finish = not fail && notElem Unknow (concat res)
        (n', fail, _, res) = checkLoop (n, if even n then initH else initV, b)
        initH = replicate (length b) False
        initV = replicate (length $ head b) False
        (rnd', bGuess) = randomChange rnd ori

main :: IO ()
main = do
  let maxIter = 100
      seed = 0.5
      initBoard = replicate (length verticalConst) $ replicate (length horizonConst) Unknow
      (n, fail, finished, b) = solve seed maxIter initBoard initBoard
  putStr $ "after iter " ++ show (maxIter - n + 1) ++ ": "
  putStrLn $ case (fail, finished) of
    (True, _) -> "Impossible!"
    (_, True) -> "Finished!"
    _ -> "Unfinished!"
  putStr $ showBoard b
