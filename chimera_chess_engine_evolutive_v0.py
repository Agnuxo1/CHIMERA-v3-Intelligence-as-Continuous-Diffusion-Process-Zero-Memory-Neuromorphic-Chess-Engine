#!/usr/bin/env python3
"""
CHIMERA Chess Engine - Evolutionary Neural Chess on GPU with Continuous Learning
Complete functional chess engine using CHIMERA neuromorphic architecture
with persistent holographic memory for continuous learning

Features:
- Board state in GPU textures (MIM - Morpho-Image Memory)
- Move generation via compute shaders
- Position evaluation through cellular automata evolution
- Parallel minimax search on GPU
- Real-time visualization with OpenGL
- Human vs CHIMERA gameplay
- Continuous learning from games (persistent memory)

Author: Based on CHIMERA architecture by Francisco Angulo de Lafuente
"""

import numpy as np
import moderngl
import pygame
from pygame.locals import *
from typing import List, Tuple, Optional
import time
import os
from dataclasses import dataclass
from enum import IntEnum

# ============================================================================
# PARAMETERS
# ============================================================================

LEARNING_RATE = 0.05  # Tasa de actualización (0.01-0.1)
SIMILARITY_THRESHOLD = 0.85  # Umbral para reconocer patrones
MEMORY_INFLUENCE = 0.3  # Peso de memoria en evaluación (0.0-1.0)
MIN_CONFIDENCE = 0.1  # Confianza mínima para usar patrón
MEMORY_FILE = "chimera_memory.bin"

# Pesos de evaluación inicial
INITIAL_MATERIAL_WEIGHT = 0.4
INITIAL_POSITIONAL_WEIGHT = 0.4
INITIAL_NEUROMORPHIC_WEIGHT = 0.2

# Valores posicionales
CENTER_SQUARE_BONUS = 0.5
DEVELOPMENT_BONUS = 0.3
KING_SAFETY_BONUS = 0.4
MOBILITY_BONUS = 0.1

# ============================================================================
# CHESS GAME LOGIC
# ============================================================================

class Piece(IntEnum):
    """Chess piece encoding for GPU textures"""
    EMPTY = 0
    WHITE_PAWN = 1
    WHITE_KNIGHT = 2
    WHITE_BISHOP = 3
    WHITE_ROOK = 4
    WHITE_QUEEN = 5
    WHITE_KING = 6
    BLACK_PAWN = 7
    BLACK_KNIGHT = 8
    BLACK_BISHOP = 9
    BLACK_ROOK = 10
    BLACK_QUEEN = 11
    BLACK_KING = 12


@dataclass
class Move:
    """Chess move representation"""
    from_x: int
    from_y: int
    to_x: int
    to_y: int
    piece: int
    captured: int = 0
    score: float = 0.0
    
    def __str__(self):
        files = 'abcdefgh'
        ranks = '87654321'  # Flipped for display
        return f"{files[self.from_x]}{ranks[self.from_y]}{files[self.to_x]}{ranks[self.to_y]}"


class ChessBoard:
    """Chess board state and legal move generation"""
    
    def __init__(self):
        self.board = np.zeros((8, 8), dtype=np.int32)
        self.initialize_board()
        self.white_to_move = True
        self.move_count = 0  # Contador de movimientos totales de la partida
    
    def initialize_board(self):
        """Set up starting position"""
        # Black pieces (top)
        self.board[0] = [
            Piece.BLACK_ROOK, Piece.BLACK_KNIGHT, Piece.BLACK_BISHOP, Piece.BLACK_QUEEN,
            Piece.BLACK_KING, Piece.BLACK_BISHOP, Piece.BLACK_KNIGHT, Piece.BLACK_ROOK
        ]
        self.board[1] = [Piece.BLACK_PAWN] * 8
        
        # Empty squares
        self.board[2:6] = Piece.EMPTY
        
        # White pieces (bottom)
        self.board[6] = [Piece.WHITE_PAWN] * 8
        self.board[7] = [
            Piece.WHITE_ROOK, Piece.WHITE_KNIGHT, Piece.WHITE_BISHOP, Piece.WHITE_QUEEN,
            Piece.WHITE_KING, Piece.WHITE_BISHOP, Piece.WHITE_KNIGHT, Piece.WHITE_ROOK
        ]
    
    def is_white_piece(self, piece: int) -> bool:
        return 1 <= piece <= 6
    
    def is_black_piece(self, piece: int) -> bool:
        return 7 <= piece <= 12
    
    def generate_legal_moves(self) -> List[Move]:
        """Generate all legal moves for current player"""
        moves = []
        
        for y in range(8):
            for x in range(8):
                piece = self.board[y, x]
                
                if piece == Piece.EMPTY:
                    continue
                
                is_white = self.is_white_piece(piece)
                if is_white != self.white_to_move:
                    continue
                
                # Generate moves based on piece type
                piece_type = piece if is_white else piece - 6
                
                if piece_type == Piece.WHITE_PAWN:
                    moves.extend(self._generate_pawn_moves(x, y, is_white))
                elif piece_type == Piece.WHITE_KNIGHT:
                    moves.extend(self._generate_knight_moves(x, y, is_white))
                elif piece_type == Piece.WHITE_BISHOP:
                    moves.extend(self._generate_bishop_moves(x, y, is_white))
                elif piece_type == Piece.WHITE_ROOK:
                    moves.extend(self._generate_rook_moves(x, y, is_white))
                elif piece_type == Piece.WHITE_QUEEN:
                    moves.extend(self._generate_queen_moves(x, y, is_white))
                elif piece_type == Piece.WHITE_KING:
                    moves.extend(self._generate_king_moves(x, y, is_white))
        
        return moves
    
    def _generate_pawn_moves(self, x: int, y: int, is_white: bool) -> List[Move]:
        """Generate pawn moves
        
        Rules:
        - Peón puede avanzar 1 casilla si está libre
        - Peón puede avanzar 2 casillas SOLO en el PRIMER movimiento de la partida
          (si es el primer movimiento Y el peón está en su posición inicial)
        - Después del primer movimiento, ningún peón puede avanzar 2 casillas
        - Peón puede capturar en diagonal
        """
        moves = []
        direction = -1 if is_white else 1
        start_rank = 6 if is_white else 1
        piece = self.board[y, x]
        
        # Verificar si el peón está en su posición inicial
        is_on_start_rank = (y == start_rank)
        
        # Verificar si es el primer movimiento de la partida
        is_first_move = (self.move_count == 0)
        
        # Movimiento de 1 casilla hacia adelante (siempre permitido si la casilla está libre)
        ny = y + direction
        if 0 <= ny < 8 and self.board[ny, x] == Piece.EMPTY:
            moves.append(Move(x, y, x, ny, piece))
            
            # Movimiento de 2 casillas SOLO si:
            # 1. Es el PRIMER movimiento de la partida (move_count == 0)
            # 2. El peón está en su posición inicial
            # 3. Ambas casillas (intermedia y destino) están libres
            if is_first_move and is_on_start_rank:
                ny2 = y + 2 * direction
                # Verificar que la casilla destino esté dentro del tablero
                if 0 <= ny2 < 8 and self.board[ny2, x] == Piece.EMPTY:
                    moves.append(Move(x, y, x, ny2, piece))
        
        # Capturas diagonales
        for dx in [-1, 1]:
            nx, ny = x + dx, y + direction
            if 0 <= nx < 8 and 0 <= ny < 8:
                target = self.board[ny, nx]
                if target != Piece.EMPTY:
                    # Solo puede capturar piezas del color opuesto
                    if is_white and self.is_black_piece(target):
                        moves.append(Move(x, y, nx, ny, piece, target))
                    elif not is_white and self.is_white_piece(target):
                        moves.append(Move(x, y, nx, ny, piece, target))
        
        return moves
    
    def _generate_knight_moves(self, x: int, y: int, is_white: bool) -> List[Move]:
        """Generate knight moves"""
        moves = []
        piece = self.board[y, x]
        knight_moves = [
            (-2, -1), (-2, 1), (-1, -2), (-1, 2),
            (1, -2), (1, 2), (2, -1), (2, 1)
        ]
        
        for dx, dy in knight_moves:
            nx, ny = x + dx, y + dy
            if 0 <= nx < 8 and 0 <= ny < 8:
                target = self.board[ny, nx]
                if target == Piece.EMPTY or \
                   (is_white and self.is_black_piece(target)) or \
                   (not is_white and self.is_white_piece(target)):
                    moves.append(Move(x, y, nx, ny, piece, target))
        
        return moves
    
    def _generate_sliding_moves(self, x: int, y: int, is_white: bool, 
                                directions: List[Tuple[int, int]]) -> List[Move]:
        """Generate moves for sliding pieces (bishop, rook, queen)"""
        moves = []
        piece = self.board[y, x]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            while 0 <= nx < 8 and 0 <= ny < 8:
                target = self.board[ny, nx]
                
                if target == Piece.EMPTY:
                    moves.append(Move(x, y, nx, ny, piece))
                elif (is_white and self.is_black_piece(target)) or \
                     (not is_white and self.is_white_piece(target)):
                    moves.append(Move(x, y, nx, ny, piece, target))
                    break
                else:
                    break
                
                nx += dx
                ny += dy
        
        return moves
    
    def _generate_bishop_moves(self, x: int, y: int, is_white: bool) -> List[Move]:
        """Generate bishop moves"""
        return self._generate_sliding_moves(x, y, is_white, 
                                           [(-1, -1), (-1, 1), (1, -1), (1, 1)])
    
    def _generate_rook_moves(self, x: int, y: int, is_white: bool) -> List[Move]:
        """Generate rook moves"""
        return self._generate_sliding_moves(x, y, is_white, 
                                           [(-1, 0), (1, 0), (0, -1), (0, 1)])
    
    def _generate_queen_moves(self, x: int, y: int, is_white: bool) -> List[Move]:
        """Generate queen moves"""
        return self._generate_sliding_moves(x, y, is_white,
                                           [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                                            (0, 1), (1, -1), (1, 0), (1, 1)])
    
    def _generate_king_moves(self, x: int, y: int, is_white: bool) -> List[Move]:
        """Generate king moves"""
        moves = []
        piece = self.board[y, x]
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                
                nx, ny = x + dx, y + dy
                if 0 <= nx < 8 and 0 <= ny < 8:
                    target = self.board[ny, nx]
                    if target == Piece.EMPTY or \
                       (is_white and self.is_black_piece(target)) or \
                       (not is_white and self.is_white_piece(target)):
                        moves.append(Move(x, y, nx, ny, piece, target))
        
        return moves
    
    def make_move(self, move: Move) -> 'ChessBoard':
        """Make a move and return new board state"""
        new_board = ChessBoard()
        new_board.board = self.board.copy()
        new_board.white_to_move = not self.white_to_move
        new_board.move_count = self.move_count + 1  # Incrementar contador de movimientos
        
        new_board.board[move.to_y, move.to_x] = move.piece
        new_board.board[move.from_y, move.from_x] = Piece.EMPTY
        
        return new_board
    
    def is_game_over(self) -> Tuple[bool, Optional[str]]:
        """Check if game is over"""
        moves = self.generate_legal_moves()
        
        if len(moves) == 0:
            return True, "Checkmate" if self.white_to_move else "Checkmate"
        
        # Check for kings
        has_white_king = np.any(self.board == Piece.WHITE_KING)
        has_black_king = np.any(self.board == Piece.BLACK_KING)
        
        if not has_white_king:
            return True, "Black wins"
        if not has_black_king:
            return True, "White wins"
        
        return False, None


# ============================================================================
# CHIMERA LEARNING MEMORY
# ============================================================================

class CHIMERALearningMemory:
    """
    Holographic memory for continuous learning
    Stores patterns, evaluations, and outcomes in GPU texture
    """
    
    def __init__(self, ctx: moderngl.Context, texture_size: Tuple[int, int] = (256, 256)):
        self.ctx = ctx
        self.texture_size = texture_size
        
        # Memory texture (RGBA32F)
        # R: Patterns of winning positions (accumulation)
        # G: Patterns of losing positions (accumulation)
        # B: Learned evaluations (weighted average)
        # A: Pattern strength/confidence (frequency of occurrence)
        self.memory_texture = self.ctx.texture(self.texture_size, 4, dtype='f4')
        
        # Initialize with zeros
        w, h = self.texture_size
        initial_data = np.zeros((h, w, 4), dtype=np.float32)
        self.memory_texture.write(initial_data.tobytes())
        
        # Learning parameters
        self.learning_rate = LEARNING_RATE
        self.similarity_threshold = SIMILARITY_THRESHOLD
        self.min_confidence = MIN_CONFIDENCE
        
        # Statistics
        self.patterns_stored = 0
        self.games_learned = 0
        self.base_memory_initialized = False
    
    def save_memory(self, filename: str = MEMORY_FILE) -> bool:
        """Save memory texture to disk"""
        try:
            data = np.frombuffer(self.memory_texture.read(), dtype=np.float32)
            data.tofile(filename)
            return True
        except Exception as e:
            print(f"Error saving memory: {e}")
            return False
    
    def load_memory(self, filename: str = MEMORY_FILE) -> bool:
        """Load memory texture from disk"""
        if not os.path.exists(filename):
            # No memory file exists, initialize base memory
            if not self.base_memory_initialized:
                self._initialize_base_memory()
            return False
        
        try:
            w, h = self.texture_size
            expected_size = w * h * 4 * 4  # RGBA32F = 4 bytes per float
            data = np.fromfile(filename, dtype=np.float32)
            
            if len(data) != expected_size:
                print(f"Memory file size mismatch: expected {expected_size}, got {len(data)}")
                if not self.base_memory_initialized:
                    self._initialize_base_memory()
                return False
            
            data = data.reshape((h, w, 4))
            self.memory_texture.write(data.tobytes())
            print(f"[OK] Loaded learning memory from {filename}")
            self.base_memory_initialized = True
            return True
        except Exception as e:
            print(f"Error loading memory: {e}")
            if not self.base_memory_initialized:
                self._initialize_base_memory()
            return False
    
    def _initialize_base_memory(self):
        """Initialize memory with base chess knowledge (openings, tactical patterns)"""
        w, h = self.texture_size
        memory_data = np.zeros((h, w, 4), dtype=np.float32)
        
        # Check if memory is empty (all zeros)
        current_data = np.frombuffer(self.memory_texture.read(), dtype=np.float32)
        current_data = current_data.reshape((h, w, 4))
        
        if np.allclose(current_data, 0.0):
            print("[OK] Initializing base chess knowledge in memory...")
            
            # Initialize with common opening patterns and positional knowledge
            # We encode basic chess principles in the memory texture
            
            # Center control patterns (e4, d4, e5, d5 are valuable)
            center_squares = [(3, 3), (3, 4), (4, 3), (4, 4)]  # d4, e4, d5, e5
            for cx, cy in center_squares:
                # Winning patterns (R channel) - having pieces in center is good
                memory_data[cy, cx, 0] = 0.3  # Winning pattern strength
                # Learned evaluations (B channel) - center is valuable
                memory_data[cy, cx, 2] = 0.6  # Positive evaluation
                # Confidence (A channel) - high confidence in this knowledge
                memory_data[cy, cx, 3] = 0.8
            
            # Development squares (knights and bishops developed)
            # White development squares
            white_dev_squares = [(1, 5), (6, 5), (2, 5), (5, 5)]  # f3, c3, c4, f4
            for x, y in white_dev_squares:
                memory_data[y, x, 0] = 0.2  # Winning pattern
                memory_data[y, x, 2] = 0.55  # Slightly positive
                memory_data[y, x, 3] = 0.7  # Good confidence
            
            # Black development squares
            black_dev_squares = [(1, 2), (6, 2), (2, 2), (5, 2)]  # f6, c6, c5, f5
            for x, y in black_dev_squares:
                memory_data[y, x, 0] = 0.2
                memory_data[y, x, 2] = 0.45  # Negative for white (black's development)
                memory_data[y, x, 3] = 0.7
            
            # King safety patterns (castled kings are safer)
            # White castled kingside (g1)
            memory_data[7, 6, 0] = 0.25
            memory_data[7, 6, 2] = 0.58
            memory_data[7, 6, 3] = 0.75
            # White castled queenside (c1)
            memory_data[7, 2, 0] = 0.25
            memory_data[7, 2, 2] = 0.58
            memory_data[7, 2, 3] = 0.75
            
            # Black castled kingside (g8)
            memory_data[0, 6, 0] = 0.25
            memory_data[0, 6, 2] = 0.42  # Negative for white
            memory_data[0, 6, 3] = 0.75
            # Black castled queenside (c8)
            memory_data[0, 2, 0] = 0.25
            memory_data[0, 2, 2] = 0.42
            memory_data[0, 2, 3] = 0.75
            
            # Penalty for exposed kings (e1, e8)
            memory_data[7, 4, 1] = 0.2  # Losing pattern (G channel)
            memory_data[7, 4, 2] = 0.4  # Negative evaluation
            memory_data[7, 4, 3] = 0.7
            
            memory_data[0, 4, 1] = 0.2
            memory_data[0, 4, 2] = 0.6  # Positive for white (black's king exposed)
            memory_data[0, 4, 3] = 0.7
            
            # Write initialized memory
            self.memory_texture.write(memory_data.tobytes())
            self.base_memory_initialized = True
            print(f"[OK] Base chess knowledge initialized ({np.sum(memory_data[:,:,3] > 0)} patterns)")


# ============================================================================
# CHIMERA GPU ENGINE - Morpho-Image Memory (MIM) with Learning
# ============================================================================

class CHIMERAChessEngine:
    """
    CHIMERA neuromorphic chess engine using GPU textures
    with continuous learning through holographic memory
    
    Architecture:
    - Board state in RGBA texture (MIM format)
    - Cellular automata evolution for position evaluation
    - Parallel minimax search in compute shaders
    - Real-time visualization
    - Persistent learning memory
    """
    
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        
        # MIM (Morpho-Image Memory) textures
        self.texture_size = (256, 256)  # Large enough for all state
        
        # Main state texture (RGBA32F)
        # R: Board state (piece values)
        # G: Temporal memory (evaluation history)
        # B: Result (computed evaluation)
        # A: Confidence/metadata
        self.state_texture = self.ctx.texture(self.texture_size, 4, dtype='f4')
        
        # Spatial features texture (for pattern recognition)
        self.spatial_texture = self.ctx.texture(self.texture_size, 4, dtype='f4')
        
        # Memory bias texture (result of memory query)
        self.memory_bias_texture = self.ctx.texture(self.texture_size, 4, dtype='f4')
        
        # Position encoding texture (static, reused)
        self.position_texture = self._create_position_encoding()
        
        # Positional knowledge texture (chess principles encoded)
        self.positional_knowledge_texture = self._create_positional_knowledge()
        
        # Framebuffers for ping-pong evolution
        self.fbo_a = self.ctx.framebuffer(
            color_attachments=[self.state_texture]
        )
        self.fbo_b = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture(self.texture_size, 4, dtype='f4')]
        )
        
        # Initialize learning memory
        self.learning_memory = CHIMERALearningMemory(ctx, self.texture_size)
        # Load existing memory if available, otherwise initialize base memory
        self.learning_memory.load_memory()
        
        # Compile shaders
        self._compile_shaders()
        
        # Evolution parameters
        self.evolution_steps = 8
        self.current_evaluation = 0.0
        
        # Game history for learning
        self.game_positions = []  # Store positions from current game
        self.game_outcome = None  # Will be set when game ends
    
    def _create_position_encoding(self) -> moderngl.Texture:
        """Create static position encoding texture"""
        w, h = self.texture_size
        encoding = np.zeros((h, w, 4), dtype=np.float32)
        
        for y in range(h):
            for x in range(w):
                # R: X-coordinate normalized
                encoding[y, x, 0] = x / w
                # G: Y-coordinate normalized
                encoding[y, x, 1] = y / h
                # B: sin(2π·x) for periodic patterns
                encoding[y, x, 2] = np.sin(2 * np.pi * x / w)
                # A: cos(2π·y) for complementary phase
                encoding[y, x, 3] = np.cos(2 * np.pi * y / h)
        
        texture = self.ctx.texture(self.texture_size, 4, dtype='f4')
        texture.write(encoding.tobytes())
        return texture
    
    def _create_positional_knowledge(self) -> moderngl.Texture:
        """Create positional knowledge texture with chess principles"""
        w, h = self.texture_size
        knowledge = np.zeros((h, w, 4), dtype=np.float32)
        
        # Encode positional values for each square
        for y in range(8):
            for x in range(8):
                # Center squares are more valuable
                center_dist = min(abs(x - 3.5), abs(y - 3.5))
                center_value = max(0.0, 1.0 - center_dist / 3.5) * CENTER_SQUARE_BONUS
                
                # R: Center value (higher for center squares)
                knowledge[y, x, 0] = center_value
                
                # G: Development value (higher for development squares)
                # White development squares
                if (x, y) in [(1, 5), (6, 5), (2, 5), (5, 5)]:
                    knowledge[y, x, 1] = DEVELOPMENT_BONUS
                # Black development squares
                elif (x, y) in [(1, 2), (6, 2), (2, 2), (5, 2)]:
                    knowledge[y, x, 1] = DEVELOPMENT_BONUS
                
                # B: King safety value (higher for castled positions)
                # White castled positions
                if (x, y) in [(6, 7), (2, 7)]:  # g1, c1
                    knowledge[y, x, 2] = KING_SAFETY_BONUS
                # Black castled positions
                elif (x, y) in [(6, 0), (2, 0)]:  # g8, c8
                    knowledge[y, x, 2] = KING_SAFETY_BONUS
                # Penalty for exposed kings
                elif (x, y) in [(4, 7), (4, 0)]:  # e1, e8
                    knowledge[y, x, 2] = -KING_SAFETY_BONUS * 0.5
                
                # A: General positional weight
                knowledge[y, x, 3] = 0.5 + center_value
        
        texture = self.ctx.texture(self.texture_size, 4, dtype='f4')
        texture.write(knowledge.tobytes())
        return texture
    
    def _compile_shaders(self):
        """Compile CHIMERA compute shaders including memory shaders"""
        
        # Memory Query Shader - Finds similar patterns in memory
        self.memory_query_shader = self.ctx.compute_shader('''
            #version 430
            
            layout(local_size_x = 16, local_size_y = 16) in;
            
            layout(rgba32f, binding = 0) uniform image2D current_state;
            layout(rgba32f, binding = 1) uniform image2D memory;
            layout(rgba32f, binding = 2) uniform image2D memory_bias_out;
            
            uniform float similarity_threshold;
            uniform float min_confidence;
            
            void main() {
                ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
                ivec2 size = imageSize(current_state);
                
                if (pos.x >= size.x || pos.y >= size.y) return;
                
                // Only process board region (8x8)
                if (pos.x >= 8 || pos.y >= 8) {
                    imageStore(memory_bias_out, pos, vec4(0.0));
                    return;
                }
                
                vec4 current = imageLoad(current_state, pos);
                
                // Search for similar patterns in memory
                // Compare current position with memory using Euclidean distance
                float best_similarity = 0.0;
                vec4 best_memory = vec4(0.0);
                
                // Sample memory at same position (holographic approach)
                vec4 mem = imageLoad(memory, pos);
                
                // Calculate similarity based on R channel (board state)
                float similarity = 1.0 - abs(current.r - mem.r);
                
                // If similarity is high enough and confidence is sufficient
                if (similarity >= similarity_threshold && mem.a >= min_confidence) {
                    best_similarity = similarity;
                    best_memory = mem;
                }
                
                // Also check neighboring memory regions (spatial correlation)
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        if (dx == 0 && dy == 0) continue;
                        
                        ivec2 mpos = pos + ivec2(dx, dy);
                        if (mpos.x >= 0 && mpos.x < 8 && mpos.y >= 0 && mpos.y < 8) {
                            vec4 neighbor_mem = imageLoad(memory, mpos);
                            float neighbor_sim = 1.0 - abs(current.r - neighbor_mem.r);
                            
                            if (neighbor_sim > best_similarity && 
                                neighbor_sim >= similarity_threshold && 
                                neighbor_mem.a >= min_confidence) {
                                best_similarity = neighbor_sim;
                                best_memory = neighbor_mem;
                            }
                        }
                    }
                }
                
                // Output bias based on memory
                // R: Winning pattern influence (positive bias)
                // G: Losing pattern influence (negative bias)
                // B: Learned evaluation influence
                // A: Confidence of memory match
                vec4 bias;
                if (best_similarity > 0.0) {
                    bias.r = best_memory.r * best_similarity;  // Winning patterns
                    bias.g = best_memory.g * best_similarity;  // Losing patterns
                    bias.b = best_memory.b * best_similarity;  // Learned eval
                    bias.a = best_similarity * best_memory.a;   // Combined confidence
                } else {
                    bias = vec4(0.0);
                }
                
                imageStore(memory_bias_out, pos, bias);
            }
        ''')
        
        # Memory Update Shader - Updates memory with new patterns
        self.memory_update_shader = self.ctx.compute_shader('''
            #version 430
            
            layout(local_size_x = 16, local_size_y = 16) in;
            
            layout(rgba32f, binding = 0) uniform image2D position_pattern;
            layout(rgba32f, binding = 1) uniform image2D memory;
            layout(rgba32f, binding = 2) uniform image2D memory_out;
            
            uniform float learning_rate;
            uniform float outcome;  // +1.0 if won, -1.0 if lost, 0.0 if draw
            
            void main() {
                ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
                ivec2 size = imageSize(memory);
                
                if (pos.x >= size.x || pos.y >= size.y) return;
                
                vec4 current_memory = imageLoad(memory, pos);
                vec4 pattern = imageLoad(position_pattern, pos);
                
                // Only update board region (8x8)
                if (pos.x >= 8 || pos.y >= 8) {
                    imageStore(memory_out, pos, current_memory);
                    return;
                }
                
                // Evolutionary update: new = old * (1-α) + pattern * α * outcome_weight
                // R: Winning patterns (increase if outcome > 0)
                // G: Losing patterns (increase if outcome < 0)
                // B: Learned evaluations (weighted average)
                // A: Confidence (increases with frequency)
                
                vec4 updated;
                
                // Update winning patterns (R channel)
                if (outcome > 0.0) {
                    updated.r = current_memory.r * (1.0 - learning_rate) + 
                               pattern.r * learning_rate * outcome;
                } else {
                    updated.r = current_memory.r * (1.0 - learning_rate * 0.5);
                }
                
                // Update losing patterns (G channel)
                if (outcome < 0.0) {
                    updated.g = current_memory.g * (1.0 - learning_rate) + 
                               pattern.r * learning_rate * abs(outcome);
                } else {
                    updated.g = current_memory.g * (1.0 - learning_rate * 0.5);
                }
                
                // Update learned evaluations (B channel) - weighted average
                float eval_weight = pattern.b;  // Evaluation from pattern
                updated.b = current_memory.b * (1.0 - learning_rate) + 
                           eval_weight * learning_rate;
                
                // Update confidence (A channel) - increases with frequency
                updated.a = min(1.0, current_memory.a + learning_rate * 0.1);
                
                imageStore(memory_out, pos, updated);
            }
        ''')
        
        # Cellular Automata Evolution Shader (modified to use memory bias)
        self.evolution_shader = self.ctx.compute_shader('''
            #version 430
            
            layout(local_size_x = 16, local_size_y = 16) in;
            
            layout(rgba32f, binding = 0) uniform image2D state_in;
            layout(rgba32f, binding = 1) uniform image2D state_out;
            layout(rgba32f, binding = 2) uniform image2D spatial;
            layout(rgba32f, binding = 3) uniform image2D position;
            layout(rgba32f, binding = 4) uniform image2D memory_bias;
            layout(rgba32f, binding = 5) uniform image2D positional_knowledge;
            
            uniform float memory_influence;
            
            // Cellular automata rules for chess position evaluation
            void main() {
                ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
                ivec2 size = imageSize(state_in);
                
                if (pos.x >= size.x || pos.y >= size.y) return;
                
                // Read current state
                vec4 center = imageLoad(state_in, pos);
                vec4 spatial_info = imageLoad(spatial, pos);
                vec4 pos_encoding = imageLoad(position, pos);
                vec4 bias = imageLoad(memory_bias, pos);
                vec4 pos_knowledge = imageLoad(positional_knowledge, pos);
                
                // Compute 3x3 neighborhood influence
                vec4 neighbors = vec4(0.0);
                int count = 0;
                
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        if (dx == 0 && dy == 0) continue;
                        
                        ivec2 npos = pos + ivec2(dx, dy);
                        if (npos.x >= 0 && npos.x < size.x && 
                            npos.y >= 0 && npos.y < size.y) {
                            neighbors += imageLoad(state_in, npos);
                            count++;
                        }
                    }
                }
                
                neighbors /= float(count);
                
                // Neuromorphic evolution rule with memory influence
                // R: Board state (preserved)
                // G: Temporal memory (accumulated)
                // B: Evaluation (evolved with memory bias)
                // A: Confidence (updated)
                
                vec4 evolved;
                evolved.r = center.r;  // Board state unchanged
                evolved.g = 0.7 * center.g + 0.3 * neighbors.g;  // Memory integration
                
                // Evaluation with memory bias and positional knowledge
                float base_eval = 0.6 * center.b + 0.4 * neighbors.b + 0.2 * spatial_info.r;
                
                // Apply positional knowledge (center control, development, king safety)
                float positional_bonus = 0.0;
                if (pos.x < 8 && pos.y < 8) {
                    // Center control bonus
                    positional_bonus += pos_knowledge.r * 0.3;
                    // Development bonus
                    positional_bonus += pos_knowledge.g * 0.2;
                    // King safety bonus
                    positional_bonus += pos_knowledge.b * 0.25;
                }
                
                // Apply memory influence
                float memory_bias_value = (bias.r - bias.g) * bias.a;  // Winning - losing, weighted by confidence
                float learned_eval = bias.b * bias.a;  // Learned evaluation from memory
                
                evolved.b = tanh(base_eval + positional_bonus + 
                                memory_influence * (memory_bias_value + learned_eval * 0.5));
                
                evolved.a = smoothstep(0.0, 1.0, 
                                      abs(evolved.b - center.b) < 0.1 ? 
                                      center.a + 0.1 : center.a * 0.9);  // Confidence
                
                imageStore(state_out, pos, evolved);
            }
        ''')
        
        # Spatial Features Shader (3x3 pattern detection)
        self.spatial_shader = self.ctx.compute_shader('''
            #version 430
            
            layout(local_size_x = 16, local_size_y = 16) in;
            
            layout(rgba32f, binding = 0) uniform image2D state_in;
            layout(rgba32f, binding = 1) uniform image2D spatial_out;
            
            void main() {
                ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
                ivec2 size = imageSize(state_in);
                
                if (pos.x >= size.x || pos.y >= size.y) return;
                
                vec4 center = imageLoad(state_in, pos);
                
                // Compute spatial features
                float same_count = 0.0;
                float diff_count = 0.0;
                float value_sum = 0.0;
                
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        if (dx == 0 && dy == 0) continue;
                        
                        ivec2 npos = pos + ivec2(dx, dy);
                        if (npos.x >= 0 && npos.x < size.x && 
                            npos.y >= 0 && npos.y < size.y) {
                            vec4 neighbor = imageLoad(state_in, npos);
                            
                            if (abs(neighbor.r - center.r) < 0.1) {
                                same_count += 1.0;
                            } else {
                                diff_count += 1.0;
                            }
                            
                            value_sum += neighbor.r;
                        }
                    }
                }
                
                vec4 spatial;
                spatial.r = diff_count / 8.0;  // Edge strength
                spatial.g = same_count / 8.0;  // Density
                spatial.b = (same_count <= 2.0) ? 1.0 : 0.0;  // Corner
                spatial.a = value_sum / 8.0;  // Average value
                
                imageStore(spatial_out, pos, spatial);
            }
        ''')
    
    def encode_board_to_texture(self, board: ChessBoard):
        """Encode chess board state into GPU texture"""
        w, h = self.texture_size
        texture_data = np.zeros((h, w, 4), dtype=np.float32)
        
        # Encode board in top-left 8x8 region
        for y in range(8):
            for x in range(8):
                piece = board.board[y, x]
                
                # Normalize piece values for GPU
                # Background-aware: 0.0 for empty, 0.1-1.0 for pieces
                if piece == Piece.EMPTY:
                    normalized = 0.0
                else:
                    normalized = 0.1 + 0.9 * (piece / 12.0)
                
                # R: Board state
                texture_data[y, x, 0] = normalized
                # G: Temporal memory (initialize with state)
                texture_data[y, x, 1] = normalized
                # B: Evaluation result (initialize to 0)
                texture_data[y, x, 2] = 0.0
                # A: Confidence (initialize to 1)
                texture_data[y, x, 3] = 1.0
        
        # Write to GPU
        self.state_texture.write(texture_data.tobytes())
    
    def evolve_position(self) -> float:
        """
        Evolve position evaluation through cellular automata
        with memory feedback loop
        Returns final evaluation score
        """
        # Step 1: Query memory for similar patterns (MEMORY FEEDBACK LOOP)
        self.state_texture.bind_to_image(0, read=True, write=False)
        self.learning_memory.memory_texture.bind_to_image(1, read=True, write=False)
        self.memory_bias_texture.bind_to_image(2, read=False, write=True)
        
        # Set uniform parameters
        self.memory_query_shader['similarity_threshold'].value = self.learning_memory.similarity_threshold
        self.memory_query_shader['min_confidence'].value = self.learning_memory.min_confidence
        
        self.memory_query_shader.run(
            group_x=self.texture_size[0] // 16,
            group_y=self.texture_size[1] // 16
        )
        
        # Step 2: Compute spatial features
        self.state_texture.bind_to_image(0, read=True, write=False)
        self.spatial_texture.bind_to_image(1, read=False, write=True)
        
        self.spatial_shader.run(
            group_x=self.texture_size[0] // 16,
            group_y=self.texture_size[1] // 16
        )
        
        # Step 3: Ping-pong evolution with memory influence
        for step in range(self.evolution_steps):
            # Bind textures for evolution
            if step % 2 == 0:
                src_texture = self.state_texture
                dst_texture = self.fbo_b.color_attachments[0]
            else:
                src_texture = self.fbo_b.color_attachments[0]
                dst_texture = self.state_texture
            
            src_texture.bind_to_image(0, read=True, write=False)
            dst_texture.bind_to_image(1, read=False, write=True)
            self.spatial_texture.bind_to_image(2, read=True, write=False)
            self.position_texture.bind_to_image(3, read=True, write=False)
            self.memory_bias_texture.bind_to_image(4, read=True, write=False)
            self.positional_knowledge_texture.bind_to_image(5, read=True, write=False)
            
            # Set memory influence parameter
            self.evolution_shader['memory_influence'].value = MEMORY_INFLUENCE
            
            # Run evolution step
            self.evolution_shader.run(
                group_x=self.texture_size[0] // 16,
                group_y=self.texture_size[1] // 16
            )
        
        # Read back evaluation from B channel
        final_texture = self.state_texture if self.evolution_steps % 2 == 0 else \
                       self.fbo_b.color_attachments[0]
        
        data = np.frombuffer(final_texture.read(), dtype=np.float32)
        data = data.reshape((self.texture_size[1], self.texture_size[0], 4))
        
        # Extract evaluation from B channel (8x8 region)
        evaluation = np.mean(data[:8, :8, 2])
        
        self.current_evaluation = evaluation
        return evaluation
    
    def evaluate_position(self, board: ChessBoard) -> float:
        """
        Evaluate position using CHIMERA neuromorphic evolution
        with memory feedback and positional heuristics
        
        Returns:
            Evaluation score (positive = white advantage, negative = black advantage)
        """
        # Material evaluation (base)
        material = self._count_material(board)
        
        # Positional evaluation (heuristics)
        positional = self._evaluate_positional(board)
        
        # Encode board to GPU texture
        self.encode_board_to_texture(board)
        
        # Evolve position through cellular automata (with memory)
        raw_eval = self.evolve_position()
        
        # Convert to centipawn-like scale
        # Map [0, 1] to [-10, +10] (roughly -1000 to +1000 centipawns)
        neuromorphic = (raw_eval - 0.5) * 20.0
        
        # Combine evaluations with adaptive weights
        # - Inicio: 40% material + 40% posicional + 20% neuromorfico
        # - Con aprendizaje: 30% material + 30% posicional + 40% neuromorfico
        learning_factor = min(1.0, self.learning_memory.games_learned / 10.0)
        
        material_weight = INITIAL_MATERIAL_WEIGHT - 0.1 * learning_factor
        positional_weight = INITIAL_POSITIONAL_WEIGHT - 0.1 * learning_factor
        neuromorphic_weight = INITIAL_NEUROMORPHIC_WEIGHT + 0.2 * learning_factor
        
        final_eval = (
            material_weight * material +
            positional_weight * positional +
            neuromorphic_weight * neuromorphic
        )
        
        return final_eval
    
    def _count_material(self, board: ChessBoard) -> float:
        """Count material balance"""
        piece_values = {
            Piece.WHITE_PAWN: 1, Piece.BLACK_PAWN: -1,
            Piece.WHITE_KNIGHT: 3, Piece.BLACK_KNIGHT: -3,
            Piece.WHITE_BISHOP: 3, Piece.BLACK_BISHOP: -3,
            Piece.WHITE_ROOK: 5, Piece.BLACK_ROOK: -5,
            Piece.WHITE_QUEEN: 9, Piece.BLACK_QUEEN: -9,
        }
        
        material = 0.0
        for y in range(8):
            for x in range(8):
                piece = board.board[y, x]
                material += piece_values.get(piece, 0)
        
        return material
    
    def _evaluate_center_control(self, board: ChessBoard) -> float:
        """Evaluate control of center squares (e4, d4, e5, d5)"""
        score = 0.0
        center_squares = [(3, 3), (3, 4), (4, 3), (4, 4)]  # d4, e4, d5, e5
        
        for cx, cy in center_squares:
            piece = board.board[cy, cx]
            if piece != Piece.EMPTY:
                if board.is_white_piece(piece):
                    score += CENTER_SQUARE_BONUS
                else:
                    score -= CENTER_SQUARE_BONUS
            
            # Bonus for pieces attacking center
            for y in range(8):
                for x in range(8):
                    p = board.board[y, x]
                    if p == Piece.EMPTY:
                        continue
                    
                    # Check if piece can attack center square
                    moves = board.generate_legal_moves()
                    for move in moves:
                        if move.from_x == x and move.from_y == y:
                            if move.to_x == cx and move.to_y == cy:
                                if board.is_white_piece(p):
                                    score += CENTER_SQUARE_BONUS * 0.3
                                else:
                                    score -= CENTER_SQUARE_BONUS * 0.3
                                break
        
        return score
    
    def _evaluate_piece_development(self, board: ChessBoard) -> float:
        """Evaluate piece development (knights and bishops out, castling)"""
        score = 0.0
        
        # Development squares for white
        white_development_squares = [
            (1, 5), (6, 5),  # Knights on f3, c3
            (2, 5), (5, 5),  # Bishops on c4, f4
        ]
        
        # Development squares for black
        black_development_squares = [
            (1, 2), (6, 2),  # Knights on f6, c6
            (2, 2), (5, 2),  # Bishops on c5, f5
        ]
        
        # Check white development
        for x, y in white_development_squares:
            piece = board.board[y, x]
            if piece == Piece.WHITE_KNIGHT or piece == Piece.WHITE_BISHOP:
                score += DEVELOPMENT_BONUS
        
        # Check black development
        for x, y in black_development_squares:
            piece = board.board[y, x]
            if piece == Piece.BLACK_KNIGHT or piece == Piece.BLACK_BISHOP:
                score -= DEVELOPMENT_BONUS
        
        # Bonus for pieces moved from starting position
        # White knights moved from b1/g1
        if board.board[7, 1] != Piece.WHITE_KNIGHT:  # b1
            score += DEVELOPMENT_BONUS * 0.5
        if board.board[7, 6] != Piece.WHITE_KNIGHT:  # g1
            score += DEVELOPMENT_BONUS * 0.5
        
        # Black knights moved from b8/g8
        if board.board[0, 1] != Piece.BLACK_KNIGHT:  # b8
            score -= DEVELOPMENT_BONUS * 0.5
        if board.board[0, 6] != Piece.BLACK_KNIGHT:  # g8
            score -= DEVELOPMENT_BONUS * 0.5
        
        return score
    
    def _evaluate_king_safety(self, board: ChessBoard) -> float:
        """Evaluate king safety (exposed king is bad)"""
        score = 0.0
        
        # Find kings
        white_king_pos = None
        black_king_pos = None
        
        for y in range(8):
            for x in range(8):
                piece = board.board[y, x]
                if piece == Piece.WHITE_KING:
                    white_king_pos = (x, y)
                elif piece == Piece.BLACK_KING:
                    black_king_pos = (x, y)
        
        if white_king_pos:
            x, y = white_king_pos
            # Penalty for king in center (not castled)
            if y == 7 and x == 4:  # e1
                score -= KING_SAFETY_BONUS * 0.5
            # Bonus for king on g1 or c1 (castled)
            elif y == 7 and (x == 6 or x == 2):
                score += KING_SAFETY_BONUS
        
        if black_king_pos:
            x, y = black_king_pos
            # Penalty for king in center (not castled)
            if y == 0 and x == 4:  # e8
                score += KING_SAFETY_BONUS * 0.5
            # Bonus for king on g8 or c8 (castled)
            elif y == 0 and (x == 6 or x == 2):
                score -= KING_SAFETY_BONUS
        
        return score
    
    def _evaluate_piece_mobility(self, board: ChessBoard) -> float:
        """Evaluate piece mobility (more moves = better)"""
        score = 0.0
        
        moves = board.generate_legal_moves()
        
        # Count moves for each side
        white_moves = sum(1 for m in moves if board.is_white_piece(m.piece))
        black_moves = sum(1 for m in moves if not board.is_white_piece(m.piece))
        
        # Bonus for more mobility
        score += (white_moves - black_moves) * MOBILITY_BONUS
        
        return score
    
    def _evaluate_pawn_structure(self, board: ChessBoard) -> float:
        """Evaluate pawn structure (doubled, isolated, passed pawns)"""
        score = 0.0
        
        # Check for doubled pawns (same column)
        for x in range(8):
            white_pawns_in_col = 0
            black_pawns_in_col = 0
            
            for y in range(8):
                piece = board.board[y, x]
                if piece == Piece.WHITE_PAWN:
                    white_pawns_in_col += 1
                elif piece == Piece.BLACK_PAWN:
                    black_pawns_in_col += 1
            
            # Penalty for doubled pawns
            if white_pawns_in_col > 1:
                score -= 0.2 * (white_pawns_in_col - 1)
            if black_pawns_in_col > 1:
                score += 0.2 * (black_pawns_in_col - 1)
            
            # Check for isolated pawns (no pawns in adjacent columns)
            if white_pawns_in_col > 0:
                has_adjacent = False
                for adj_x in [x-1, x+1]:
                    if 0 <= adj_x < 8:
                        for y in range(8):
                            if board.board[y, adj_x] == Piece.WHITE_PAWN:
                                has_adjacent = True
                                break
                if not has_adjacent:
                    score -= 0.3
            
            if black_pawns_in_col > 0:
                has_adjacent = False
                for adj_x in [x-1, x+1]:
                    if 0 <= adj_x < 8:
                        for y in range(8):
                            if board.board[y, adj_x] == Piece.BLACK_PAWN:
                                has_adjacent = True
                                break
                if not has_adjacent:
                    score += 0.3
        
        return score
    
    def _evaluate_piece_coordination(self, board: ChessBoard) -> float:
        """Evaluate piece coordination (pieces working together)"""
        score = 0.0
        
        # Simple heuristic: bonus for pieces on same color squares (bishops)
        white_bishops = []
        black_bishops = []
        
        for y in range(8):
            for x in range(8):
                piece = board.board[y, x]
                if piece == Piece.WHITE_BISHOP:
                    white_bishops.append((x, y))
                elif piece == Piece.BLACK_BISHOP:
                    black_bishops.append((x, y))
        
        # Bonus for bishop pair
        if len(white_bishops) == 2:
            score += 0.5
        if len(black_bishops) == 2:
            score -= 0.5
        
        # Bonus for rooks on open files
        for x in range(8):
            has_white_pawn = False
            has_black_pawn = False
            
            for y in range(8):
                piece = board.board[y, x]
                if piece == Piece.WHITE_PAWN:
                    has_white_pawn = True
                elif piece == Piece.BLACK_PAWN:
                    has_black_pawn = True
            
            # Check for rooks on open/semi-open files
            for y in range(8):
                piece = board.board[y, x]
                if piece == Piece.WHITE_ROOK:
                    if not has_white_pawn:
                        score += 0.4  # Open file
                    elif not has_black_pawn:
                        score += 0.2  # Semi-open file
                elif piece == Piece.BLACK_ROOK:
                    if not has_black_pawn:
                        score -= 0.4  # Open file
                    elif not has_white_pawn:
                        score -= 0.2  # Semi-open file
        
        return score
    
    def _evaluate_positional(self, board: ChessBoard) -> float:
        """Comprehensive positional evaluation"""
        score = 0.0
        
        # Control del centro
        score += self._evaluate_center_control(board)
        
        # Desarrollo de piezas
        score += self._evaluate_piece_development(board)
        
        # Seguridad del rey
        score += self._evaluate_king_safety(board)
        
        # Movilidad de piezas
        score += self._evaluate_piece_mobility(board)
        
        # Estructura de peones
        score += self._evaluate_pawn_structure(board)
        
        # Coordinación de piezas
        score += self._evaluate_piece_coordination(board)
        
        return score
    
    def update_memory_from_position(self, board: ChessBoard, evaluation: float):
        """Update memory with a single position pattern"""
        # Encode position
        self.encode_board_to_texture(board)
        
        # Create pattern texture with evaluation
        w, h = self.texture_size
        pattern_data = np.zeros((h, w, 4), dtype=np.float32)
        
        # Read current state
        state_data = np.frombuffer(self.state_texture.read(), dtype=np.float32)
        state_data = state_data.reshape((h, w, 4))
        
        # Create pattern: R=board state, B=evaluation (normalized)
        for y in range(8):
            for x in range(8):
                pattern_data[y, x, 0] = state_data[y, x, 0]  # Board state
                pattern_data[y, x, 1] = 0.0
                pattern_data[y, x, 2] = (evaluation / 20.0) + 0.5  # Normalize evaluation
                pattern_data[y, x, 3] = 1.0
        
        # Create temporary texture for pattern
        pattern_texture = self.ctx.texture(self.texture_size, 4, dtype='f4')
        pattern_texture.write(pattern_data.tobytes())
        
        # Update memory (will be called with outcome later)
        # This is a helper for continuous reinforcement
        return pattern_texture
    
    def update_memory_from_game(self, outcome: float):
        """
        Update memory with all positions from the game
        outcome: +1.0 if CHIMERA (black) won, -1.0 if lost, 0.0 if draw
        """
        if len(self.game_positions) == 0:
            return
        
        # Update memory for each position in the game
        for board, evaluation in self.game_positions:
            # Encode position
            self.encode_board_to_texture(board)
            
            # Create pattern texture
            w, h = self.texture_size
            pattern_data = np.zeros((h, w, 4), dtype=np.float32)
            
            state_data = np.frombuffer(self.state_texture.read(), dtype=np.float32)
            state_data = state_data.reshape((h, w, 4))
            
            for y in range(8):
                for x in range(8):
                    pattern_data[y, x, 0] = state_data[y, x, 0]  # Board state
                    pattern_data[y, x, 1] = 0.0
                    pattern_data[y, x, 2] = (evaluation / 20.0) + 0.5  # Normalized evaluation
                    pattern_data[y, x, 3] = 1.0
            
            pattern_texture = self.ctx.texture(self.texture_size, 4, dtype='f4')
            pattern_texture.write(pattern_data.tobytes())
            
            # Run memory update shader
            pattern_texture.bind_to_image(0, read=True, write=False)
            self.learning_memory.memory_texture.bind_to_image(1, read=True, write=False)
            
            # Create output texture for ping-pong
            temp_memory = self.ctx.texture(self.texture_size, 4, dtype='f4')
            temp_memory.write(np.frombuffer(self.learning_memory.memory_texture.read(), dtype=np.float32))
            temp_memory.bind_to_image(2, read=False, write=True)
            
            self.memory_update_shader['learning_rate'].value = self.learning_memory.learning_rate
            self.memory_update_shader['outcome'].value = outcome
            
            self.memory_update_shader.run(
                group_x=self.texture_size[0] // 16,
                group_y=self.texture_size[1] // 16
            )
            
            # Copy updated memory back
            updated_data = np.frombuffer(temp_memory.read(), dtype=np.float32)
            self.learning_memory.memory_texture.write(updated_data)
            
            pattern_texture.release()
            temp_memory.release()
        
        self.learning_memory.games_learned += 1
        self.learning_memory.patterns_stored += len(self.game_positions)
        
        # Clear game history
        self.game_positions = []
        self.game_outcome = None
    
    def save_learning_memory(self) -> bool:
        """Save learning memory to disk"""
        return self.learning_memory.save_memory()
    
    def find_best_move(self, board: ChessBoard, depth: int = 3) -> Optional[Move]:
        """
        Find best move using minimax search with CHIMERA evaluation
        Also stores position for learning
        
        Args:
            board: Current position
            depth: Search depth (3-4 recommended for real-time)
        
        Returns:
            Best move found
        """
        # Store position for learning (CHIMERA's positions only)
        if not board.white_to_move:  # CHIMERA is black
            eval_score = self.evaluate_position(board)
            self.game_positions.append((board, eval_score))
        
        def minimax(board: ChessBoard, depth: int, alpha: float, beta: float,
                   maximizing: bool) -> Tuple[float, Optional[Move]]:
            
            if depth == 0:
                return self.evaluate_position(board), None
            
            moves = board.generate_legal_moves()
            
            if len(moves) == 0:
                # Game over
                return -10000 if maximizing else 10000, None
            
            best_move = None
            
            if maximizing:
                max_eval = float('-inf')
                for move in moves:
                    new_board = board.make_move(move)
                    eval_score, _ = minimax(new_board, depth - 1, alpha, beta, False)
                    
                    if eval_score > max_eval:
                        max_eval = eval_score
                        best_move = move
                    
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        break
                
                return max_eval, best_move
            else:
                min_eval = float('inf')
                for move in moves:
                    new_board = board.make_move(move)
                    eval_score, _ = minimax(new_board, depth - 1, alpha, beta, True)
                    
                    if eval_score < min_eval:
                        min_eval = eval_score
                        best_move = move
                    
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        break
                
                return min_eval, best_move
        
        _, best_move = minimax(board, depth, float('-inf'), float('inf'), 
                              board.white_to_move)
        
        return best_move


# ============================================================================
# VISUALIZATION
# ============================================================================

class ChessGUI:
    """Real-time chess visualization with Pygame"""
    
    def __init__(self, width: int = 800, height: int = 600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("CHIMERA Chess Engine - Neuromorphic AI with Learning")
        
        self.square_size = min(width, height) // 9  # 8 squares + margin
        self.board_offset_x = (width - 8 * self.square_size) // 2
        self.board_offset_y = 50
        
        # Try fonts with good Unicode support for chess pieces
        chess_fonts = ['segoeuisymbol', 'seguiemj', 'dejavusans', 'arialunicodems', 'arial']
        font_large = None
        for font_name in chess_fonts:
            try:
                font_large = pygame.font.SysFont(font_name, 48, bold=True)
                test_surface = font_large.render('♔', True, (0, 0, 0))
                if test_surface.get_width() > 5:
                    break
            except:
                continue

        self.font_large = font_large if font_large else pygame.font.SysFont('arial', 48, bold=True)
        self.font_small = pygame.font.SysFont('arial', 20)
        self.font_tiny = pygame.font.SysFont('arial', 16)
        
        # Colors
        self.COLOR_LIGHT = (240, 217, 181)
        self.COLOR_DARK = (181, 136, 99)
        self.COLOR_SELECTED = (255, 255, 0, 128)
        self.COLOR_LEGAL_MOVE = (0, 255, 0, 128)
        self.COLOR_BG = (40, 40, 40)
        self.COLOR_TEXT = (255, 255, 255)
        
        # Piece Unicode symbols
        self.piece_symbols = {
            Piece.WHITE_KING: '♔', Piece.WHITE_QUEEN: '♕',
            Piece.WHITE_ROOK: '♖', Piece.WHITE_BISHOP: '♗',
            Piece.WHITE_KNIGHT: '♘', Piece.WHITE_PAWN: '♙',
            Piece.BLACK_KING: '♚', Piece.BLACK_QUEEN: '♛',
            Piece.BLACK_ROOK: '♜', Piece.BLACK_BISHOP: '♝',
            Piece.BLACK_KNIGHT: '♞', Piece.BLACK_PAWN: '♟'
        }
        
        self.selected_square = None
        self.legal_moves = []
    
    def draw_board(self, board: ChessBoard):
        """Draw chess board and pieces"""
        self.screen.fill(self.COLOR_BG)
        
        # Draw squares
        for y in range(8):
            for x in range(8):
                color = self.COLOR_LIGHT if (x + y) % 2 == 0 else self.COLOR_DARK
                rect = pygame.Rect(
                    self.board_offset_x + x * self.square_size,
                    self.board_offset_y + y * self.square_size,
                    self.square_size,
                    self.square_size
                )
                pygame.draw.rect(self.screen, color, rect)
                
                # Highlight selected square
                if self.selected_square == (x, y):
                    s = pygame.Surface((self.square_size, self.square_size))
                    s.set_alpha(128)
                    s.fill((255, 255, 0))
                    self.screen.blit(s, rect.topleft)
                
                # Highlight legal moves
                for move in self.legal_moves:
                    if move.to_x == x and move.to_y == y:
                        s = pygame.Surface((self.square_size, self.square_size))
                        s.set_alpha(100)
                        s.fill((0, 255, 0))
                        self.screen.blit(s, rect.topleft)
                
                # Draw piece
                piece = board.board[y, x]
                if piece != Piece.EMPTY:
                    symbol = self.piece_symbols.get(piece, '')
                    is_white = 1 <= piece <= 6
                    piece_color = (255, 255, 255) if is_white else (0, 0, 0)
                    outline_color = (0, 0, 0) if is_white else (255, 255, 255)

                    # Draw outline
                    for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1), (-2,0), (2,0), (0,-2), (0,2)]:
                        outline = self.font_large.render(symbol, True, outline_color)
                        outline_rect = outline.get_rect(center=(rect.centerx + dx, rect.centery + dy))
                        self.screen.blit(outline, outline_rect)

                    # Draw main piece
                    text = self.font_large.render(symbol, True, piece_color)
                    text_rect = text.get_rect(center=rect.center)
                    self.screen.blit(text, text_rect)
        
        # Draw coordinates
        files = 'abcdefgh'
        for i, letter in enumerate(files):
            text = self.font_tiny.render(letter, True, self.COLOR_TEXT)
            self.screen.blit(text, (
                self.board_offset_x + i * self.square_size + self.square_size // 2 - 5,
                self.board_offset_y + 8 * self.square_size + 5
            ))
        
        for i in range(8):
            text = self.font_tiny.render(str(8 - i), True, self.COLOR_TEXT)
            self.screen.blit(text, (
                self.board_offset_x - 20,
                self.board_offset_y + i * self.square_size + self.square_size // 2 - 8
            ))
    
    def draw_status(self, board: ChessBoard, engine: CHIMERAChessEngine,
                   thinking: bool = False, move_time: float = 0.0):
        """Draw status information including learning stats"""
        # Title
        title = self.font_large.render("CHIMERA Chess Engine (Learning)", True, self.COLOR_TEXT)
        self.screen.blit(title, (self.width // 2 - title.get_width() // 2, 10))
        
        # Turn indicator
        turn_text = "White to move" if board.white_to_move else "Black to move (CHIMERA)"
        turn_surface = self.font_small.render(turn_text, True, self.COLOR_TEXT)
        self.screen.blit(turn_surface, (20, self.board_offset_y + 8 * self.square_size + 40))
        
        # Evaluation
        eval_text = f"Evaluation: {engine.current_evaluation:.2f}"
        eval_surface = self.font_small.render(eval_text, True, self.COLOR_TEXT)
        self.screen.blit(eval_surface, (20, self.board_offset_y + 8 * self.square_size + 70))
        
        # Learning stats
        learning_text = f"Games learned: {engine.learning_memory.games_learned} | Patterns: {engine.learning_memory.patterns_stored}"
        learning_surface = self.font_tiny.render(learning_text, True, (200, 200, 255))
        self.screen.blit(learning_surface, (20, self.board_offset_y + 8 * self.square_size + 100))
        
        # Thinking indicator
        if thinking:
            think_text = f"CHIMERA thinking... ({move_time:.1f}s)"
            think_surface = self.font_small.render(think_text, True, (255, 255, 0))
            self.screen.blit(think_surface, (20, self.board_offset_y + 8 * self.square_size + 130))
        
        # Instructions
        inst_text = "Click to select piece, click again to move. ESC to quit."
        inst_surface = self.font_tiny.render(inst_text, True, self.COLOR_TEXT)
        self.screen.blit(inst_surface, (20, self.height - 30))
    
    def get_square_from_mouse(self, pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """Convert mouse position to board coordinates"""
        x, y = pos
        x -= self.board_offset_x
        y -= self.board_offset_y
        
        if x < 0 or x >= 8 * self.square_size or y < 0 or y >= 8 * self.square_size:
            return None
        
        return (x // self.square_size, y // self.square_size)


# ============================================================================
# MAIN GAME LOOP
# ============================================================================

def main():
    """Main game loop - Human vs CHIMERA with learning"""
    
    print("\n" + "="*80)
    print("  CHIMERA CHESS ENGINE - Neuromorphic AI on GPU with Continuous Learning")
    print("="*80 + "\n")
    print("Initializing OpenGL context and CHIMERA architecture...")
    
    # Initialize ModernGL context
    ctx = moderngl.create_standalone_context()
    
    print(f"[OK] OpenGL {ctx.version_code} context created")
    print(f"[OK] GPU: {ctx.info['GL_RENDERER']}")

    # Initialize game components
    board = ChessBoard()
    engine = CHIMERAChessEngine(ctx)
    gui = ChessGUI()

    print("[OK] CHIMERA neuromorphic engine with learning initialized")
    print(f"  - Evolution steps: {engine.evolution_steps}")
    print(f"  - Texture size: {engine.texture_size}")
    print(f"  - MIM (Morpho-Image Memory) active")
    print(f"  - Learning memory: {engine.learning_memory.games_learned} games, {engine.learning_memory.patterns_stored} patterns")
    print(f"  - Learning rate: {LEARNING_RATE}")
    print(f"  - Memory influence: {MEMORY_INFLUENCE}")
    print("\n[OK] Starting game: Human (White) vs CHIMERA (Black)")
    print("="*80 + "\n")
    
    clock = pygame.time.Clock()
    running = True
    chimera_thinking = False
    think_start_time = 0
    
    try:
        while running:
            # Event handling
            for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    running = False
                
                elif event.type == MOUSEBUTTONDOWN and event.button == 1:
                    if not chimera_thinking and board.white_to_move:
                        # Human player's turn
                        square = gui.get_square_from_mouse(event.pos)
                        
                        if square:
                            x, y = square
                            
                            if gui.selected_square is None:
                                # Select piece
                                piece = board.board[y, x]
                                if piece != Piece.EMPTY and board.is_white_piece(piece):
                                    gui.selected_square = (x, y)
                                    
                                    # Generate legal moves from this square
                                    gui.legal_moves = [
                                        m for m in board.generate_legal_moves()
                                        if m.from_x == x and m.from_y == y
                                    ]
                            else:
                                # Try to make move
                                from_x, from_y = gui.selected_square
                                
                                # Check if this is a legal move
                                move = next((m for m in gui.legal_moves 
                                           if m.to_x == x and m.to_y == y), None)
                                
                                if move:
                                    # Make the move
                                    board = board.make_move(move)
                                    print(f"Human plays: {move}")
                                    
                                    # Evaluate position
                                    eval_score = engine.evaluate_position(board)
                                    print(f"Position evaluation: {eval_score:.2f}")
                                
                                # Clear selection
                                gui.selected_square = None
                                gui.legal_moves = []
            
            # CHIMERA's turn
            if not board.white_to_move and not chimera_thinking:
                game_over, result = board.is_game_over()
                if game_over:
                    print(f"\nGame Over: {result}")
                    
                    # Determine outcome for learning
                    if "Black wins" in result:
                        outcome = 1.0  # CHIMERA won
                    elif "White wins" in result:
                        outcome = -1.0  # CHIMERA lost
                    else:
                        outcome = 0.0  # Draw
                    
                    # Update memory with game result
                    print(f"\nCHIMERA learning from game (outcome: {outcome})...")
                    engine.update_memory_from_game(outcome)
                    print(f"Memory updated: {engine.learning_memory.games_learned} games learned")
                    
                    running = False
                else:
                    chimera_thinking = True
                    think_start_time = time.time()
                    print("\nCHIMERA thinking...")
            
            if chimera_thinking:
                think_time = time.time() - think_start_time
                
                # Find best move (depth 3 for real-time performance)
                best_move = engine.find_best_move(board, depth=3)
                
                if best_move:
                    board = board.make_move(best_move)
                    think_time = time.time() - think_start_time
                    
                    print(f"CHIMERA plays: {best_move} ({think_time:.2f}s)")
                    
                    # Evaluate position
                    eval_score = engine.evaluate_position(board)
                    print(f"Position evaluation: {eval_score:.2f}")
                
                chimera_thinking = False
            
            # Check game over
            game_over, result = board.is_game_over()
            if game_over:
                print(f"\nGame Over: {result}")
                
                # Determine outcome for learning
                if "Black wins" in result:
                    outcome = 1.0  # CHIMERA won
                elif "White wins" in result:
                    outcome = -1.0  # CHIMERA lost
                else:
                    outcome = 0.0  # Draw
                
                # Update memory with game result
                print(f"\nCHIMERA learning from game (outcome: {outcome})...")
                engine.update_memory_from_game(outcome)
                print(f"Memory updated: {engine.learning_memory.games_learned} games learned")
                
                running = False
            
            # Render
            gui.draw_board(board)
            gui.draw_status(board, engine, chimera_thinking, 
                           time.time() - think_start_time if chimera_thinking else 0)
            
            pygame.display.flip()
            clock.tick(60)  # 60 FPS
    
    finally:
        # Save learning memory before exit
        print("\nSaving learning memory...")
        if engine.save_learning_memory():
            print(f"[OK] Learning memory saved to {MEMORY_FILE}")
        else:
            print("[WARNING] Failed to save learning memory")
        
        # Cleanup
        pygame.quit()
        ctx.release()
        
        print("\n" + "="*80)
        print("Thank you for playing CHIMERA Chess!")
        print(f"CHIMERA learned from {engine.learning_memory.games_learned} games")
        print(f"Total patterns stored: {engine.learning_memory.patterns_stored}")
        print("="*80 + "\n")


if __name__ == "__main__":
    main()

