import os
import random
from PIL import Image, ImageDraw, ImageFont

def get_valid_ticket_grid():
    """Generates a valid 3x9 grid layout (15 numbers, 5 per row, 1-3 per column)."""
    while True:
        grid = [[0] * 9 for _ in range(3)]
        
        # Guarantee each column gets at least 1 number
        for col in range(9):
            grid[random.randint(0, 2)][col] = 1
            
        candidates = [(r, c) for r in range(3) for c in range(9) if grid[r][c] == 0]
        random.shuffle(candidates)
        
        # Backtrack to place remaining 6 numbers
        def backtrack(idx, count_added):
            if count_added == 6:
                return (all(sum(grid[r]) == 5 for r in range(3)) and 
                        all(1 <= sum(grid[r][c] for r in range(3)) <= 3 for c in range(9)))
                
            if idx >= len(candidates):
                return False
                
            r, c = candidates[idx]
            if sum(grid[r]) < 5 and sum(grid[r2][c] for r2 in range(3)) < 3:
                grid[r][c] = 1
                if backtrack(idx + 1, count_added + 1):
                    return True
                grid[r][c] = 0
                
            return backtrack(idx + 1, count_added)
            
        if backtrack(0, 0):
            return grid

def generate_single_housie_ticket():
    """Generates a single Housie ticket populated with values adhering to column ranges and sorting."""
    grid = get_valid_ticket_grid()
    ticket = [[None] * 9 for _ in range(3)]
    
    # Column ranges (1-9, 10-19, ..., 80-90)
    col_ranges = [
        (1, 9), (10, 19), (20, 29), (30, 39), (40, 49),
        (50, 59), (60, 69), (70, 79), (80, 90)
    ]
    
    # Assign sorted numbers into active grid spots
    for c in range(9):
        active_rows = [r for r in range(3) if grid[r][c] == 1]
        low, high = col_ranges[c]
        nums = sorted(random.sample(range(low, high + 1), len(active_rows)))
        for idx, r in enumerate(active_rows):
            ticket[r][c] = nums[idx]
            
    return ticket

def validate_housie_ticket(ticket):
    """Validates ticket structure, counts, number ranges, and ordering."""
    assert len(ticket) == 3 and all(len(row) == 9 for row in ticket)
    
    # Check 5 numbers per row
    for r in range(3):
        assert sum(1 for val in ticket[r] if val is not None) == 5
        
    # Check column counts, ranges, and ascending order
    col_ranges = [
        (1, 9), (10, 19), (20, 29), (30, 39), (40, 49),
        (50, 59), (60, 69), (70, 79), (80, 90)
    ]
    all_numbers = []
    for c in range(9):
        col_vals = [ticket[r][c] for r in range(3) if ticket[r][c] is not None]
        assert 1 <= len(col_vals) <= 3
        low, high = col_ranges[c]
        for val in col_vals:
            assert low <= val <= high
            all_numbers.append(val)
        assert col_vals == sorted(col_vals)
        
    assert len(all_numbers) == 15 and len(all_numbers) == len(set(all_numbers))

def render_tickets_image(tickets, output_filepath=os.path.join("Output_images", "housie_tickets.png")):
    """Renders all generated Housie tickets into a single PNG image in Output_images folder."""
    output_dir = os.path.dirname(output_filepath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    cell_width, cell_height = 80, 70
    grid_width, grid_height = cell_width * 9, cell_height * 3
    ticket_header_height, ticket_padding_v = 45, 25
    ticket_total_height = ticket_header_height + grid_height + ticket_padding_v
    
    margin_x, margin_top, margin_bottom = 50, 100, 50
    image_width = grid_width + (margin_x * 2)
    image_height = margin_top + (len(tickets) * ticket_total_height) + margin_bottom
    
    # Styling colors
    bg_color = (245, 247, 250)
    card_bg = (255, 255, 255)
    empty_cell_bg = (238, 242, 246)
    header_colors = [(41, 128, 185), (39, 174, 96), (142, 68, 173), (211, 84, 0), (192, 57, 43)]
    border_color = (180, 190, 200)
    grid_line_color = (200, 210, 220)
    text_color = (30, 40, 50)
    title_color = (24, 43, 73)
    header_text_color = (255, 255, 255)
    
    image = Image.new("RGB", (image_width, image_height), bg_color)
    draw = ImageDraw.Draw(image)
    
    try:
        title_font = ImageFont.truetype("arial.ttf", 32)
        subtitle_font = ImageFont.truetype("arial.ttf", 16)
        header_font = ImageFont.truetype("arial.ttf", 20)
        number_font = ImageFont.truetype("arialbd.ttf", 30)
    except IOError:
        title_font = subtitle_font = header_font = number_font = ImageFont.load_default()

    # Draw header title
    title_text = "HOUSIE / TAMBOLA TICKETS"
    title_bbox = draw.textbbox((0, 0), title_text, font=title_font)
    draw.text(((image_width - (title_bbox[2] - title_bbox[0])) // 2, 25), title_text, fill=title_color, font=title_font)
    
    sub_text = "Standard 3x9 Tickets • 15 Numbers Per Ticket • 5 Numbers Per Row"
    sub_bbox = draw.textbbox((0, 0), sub_text, font=subtitle_font)
    draw.text(((image_width - (sub_bbox[2] - sub_bbox[0])) // 2, 65), sub_text, fill=(100, 115, 130), font=subtitle_font)

    # Render tickets
    for idx, ticket in enumerate(tickets):
        top_y = margin_top + (idx * ticket_total_height)
        left_x = margin_x
        header_bg = header_colors[idx % len(header_colors)]
        
        # Ticket header bar
        draw.rectangle([left_x, top_y, left_x + grid_width, top_y + ticket_header_height], fill=header_bg)
        ticket_title = f"TICKET #{idx + 1}"
        h_bbox = draw.textbbox((0, 0), ticket_title, font=header_font)
        draw.text((left_x + 15, top_y + (ticket_header_height - (h_bbox[3] - h_bbox[1])) // 2), ticket_title, fill=header_text_color, font=header_font)

        # Ticket grid cells
        grid_top_y = top_y + ticket_header_height
        for r in range(3):
            for c in range(9):
                cell_x1, cell_y1 = left_x + (c * cell_width), grid_top_y + (r * cell_height)
                cell_x2, cell_y2 = cell_x1 + cell_width, cell_y1 + cell_height
                val = ticket[r][c]
                
                if val is None:
                    draw.rectangle([cell_x1, cell_y1, cell_x2, cell_y2], fill=empty_cell_bg, outline=grid_line_color, width=1)
                else:
                    draw.rectangle([cell_x1, cell_y1, cell_x2, cell_y2], fill=card_bg, outline=grid_line_color, width=1)
                    val_str = str(val)
                    bbox = draw.textbbox((0, 0), val_str, font=number_font)
                    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    draw.text((cell_x1 + (cell_width - w) // 2, cell_y1 + (cell_height - h) // 2 - 3), val_str, fill=text_color, font=number_font)
                    
        draw.rectangle([left_x, top_y, left_x + grid_width, grid_top_y + grid_height], outline=border_color, width=2)
        
    image.save(output_filepath)
    print(f"Successfully rendered 5 Housie tickets to: '{output_filepath}'")

def main():
    tickets = []
    for i in range(5):
        ticket = generate_single_housie_ticket()
        validate_housie_ticket(ticket)
        tickets.append(ticket)
        
    output_path = os.path.join("Output_images", "housie_tickets.png")
    render_tickets_image(tickets, output_path)

if __name__ == "__main__":
    main()
