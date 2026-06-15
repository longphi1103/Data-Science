import asyncio
import random
import os
import re
import pandas as pd
from playwright.async_api import async_playwright

# ==========================================
# CẤU HÌNH GIAI ĐOẠN 2
# ==========================================
INPUT_FILE = "danh_sach_link.txt"     
START_INDEX = 245
END_INDEX = 246

MAX_TOTAL_REVIEWS = 50         
MIN_LENGTH = 30                       
MAX_LENGTH = 1000                     

SUMMARY_FILE = "thong_ke_nha_hang1.csv"   
# Đổi đuôi thành CSV để lưu cả Text và Sao cho chuẩn form
REVIEWS_FILE = "du_lieu_reviews_kem_sao.csv"    

async def human_delay(a=1.2, b=2.5):
    await asyncio.sleep(random.uniform(a, b))

async def change_sort(page, sort_type="Lowest"):
    try:
        sort_btn = await page.wait_for_selector(
            "button:has-text('Phù hợp nhất'), button:has-text('Mới nhất'), button:has-text('Xếp hạng thấp nhất'), button:has-text('Xếp hạng cao nhất'), button[aria-label*='Sắp xếp']", 
            timeout=5000
        )
        await sort_btn.scroll_into_view_if_needed()
        await sort_btn.click()
        await human_delay(1.5, 2.5) 
        
        if sort_type == "Lowest":
            option_locator = page.locator('div[role="menuitemradio"]:has-text("Xếp hạng thấp nhất")')
        else:
            option_locator = page.locator('div[role="menuitemradio"]:has-text("Xếp hạng cao nhất")')
            
        await option_locator.wait_for(state="visible", timeout=5000)
        await option_locator.evaluate("node => node.click()")
        await human_delay(2.5, 4) 
        return True
    except Exception as e:
        print(f"Lỗi đổi bộ lọc: {e}") 
        return False

async def scrape_current_view(page, current_reviews_dict, limit):
    """Cào cho đến khi TỔNG giỏ reviews đạt 'limit'. Dùng Dict {text: sao} để lọc trùng."""
    last_count = 0
    same_count = 0
    
    try:
        await page.wait_for_selector("div.jftiEf", timeout=7000)
    except: return 0

    start_count = len(current_reviews_dict)

    while len(current_reviews_dict) < limit:
        blocks = await page.query_selector_all("div.jftiEf")
        for b in blocks:
            try:
                # --- 1. LẤY SỐ SAO ---
                star_rating = ""
                star_el = await b.query_selector("span[role='img']")
                if star_el:
                    aria_label = await star_el.get_attribute("aria-label")
                    if aria_label:
                        match = re.search(r'\d+', aria_label)
                        if match:
                            star_rating = match.group()

                # --- 2. BẤM NÚT DỊCH (TRANSLATE) NẾU CÓ ---
                # Tìm tất cả các nút có thể là nút dịch
                translate_btn = await b.query_selector("button:has-text('Dịch'), button:has-text('Translate')")
                if translate_btn:
                    btn_text = await translate_btn.inner_text()
                    # Chỉ bấm nếu không phải là nút "Bản gốc" hay "Original"
                    if "bản gốc" not in btn_text.lower() and "original" not in btn_text.lower():
                        try:
                            await translate_btn.click()
                            await human_delay(0.8, 1.5) # Chờ Google load API dịch
                        except: pass

                # --- 3. LẤY TEXT REVIEW ---
                review_container = await b.query_selector("div.MyEned")
                if not review_container: continue

                # Bấm xem thêm nếu có
                more_btn = await review_container.query_selector("button:has-text('Xem thêm'), button:has-text('More')")
                if more_btn:
                    try:
                        await more_btn.click()
                        await human_delay(0.4, 0.7)
                    except: pass

                text_el = await review_container.query_selector("span.wiI7pd")
                if text_el:
                    review_text = await text_el.inner_text()
                    if MIN_LENGTH <= len(review_text) <= MAX_LENGTH:
                        clean_text = review_text.replace('\n', ' ').strip()
                        
                        # Lưu vào Dict: key là text (để tự động lọc trùng), value là số sao
                        current_reviews_dict[clean_text] = star_rating
                        
                        if len(current_reviews_dict) >= limit: break
            except: continue

        if len(current_reviews_dict) >= limit: break

        # --- CUỘN TRANG ---
        if blocks:
            await blocks[-1].hover() # Đưa chuột vào review cuối để ép cuộn đúng khung
        await page.mouse.wheel(0, 1500)
        
        await human_delay(2, 3)

        current_on_screen = len(await page.query_selector_all("div.jftiEf"))
        if current_on_screen == last_count:
            same_count += 1
            if same_count > 3: break
        else: same_count = 0
        last_count = current_on_screen
    
    return len(current_reviews_dict) - start_count

async def run():
    if not os.path.exists(INPUT_FILE): return
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        urls_list = [line.strip() for line in f if line.strip()]
    ca_hien_tai = urls_list[START_INDEX:END_INDEX]
    
    async with async_playwright() as p:
        browser = await p.chromium.launch_persistent_context(
            user_data_dir=f"user_data_reviews_{START_INDEX}",
            headless=False,
            args=["--start-maximized", "--disable-blink-features=AutomationControlled"],
            locale="vi-VN"
        )
        page = await browser.new_page()

        for idx, url in enumerate(ca_hien_tai):
            print(f"\n[{idx+1}/{len(ca_hien_tai)}] Quán số {START_INDEX + idx}")
            try:
                await page.goto(url)
                await human_delay(4, 6)
                
                name_el = await page.wait_for_selector("h1", timeout=5000)
                restaurant_name = await name_el.inner_text()
                
                btn = await page.wait_for_selector("//button[contains(., 'Bài đánh giá') or contains(., 'Reviews')]", timeout=5000)
                await btn.click()
                await human_delay(2, 4)
                
                # Khởi tạo DICT
                final_reviews_dict = {}

                # Tính toán số lượng cần lấy cho một nửa
                HALF_LIMIT = MAX_TOTAL_REVIEWS // 2

                # BƯỚC 1: Lấy đúng 1 nửa là đánh giá Xấu
                print(f"⏬ Đang gom {HALF_LIMIT} đánh giá Xấu...")
                bad_count = 0
                if await change_sort(page, "Lowest"):
                    # Giới hạn giỏ chỉ lấy đến HALF_LIMIT (VD: 20)
                    bad_count = await scrape_current_view(page, final_reviews_dict, HALF_LIMIT)
                
                # BƯỚC 2: Lấy đánh giá Tốt để lấp đầy giỏ
                good_count = 0
                if len(final_reviews_dict) < MAX_TOTAL_REVIEWS:
                    remaining = MAX_TOTAL_REVIEWS - len(final_reviews_dict)
                    print(f"🔄 Đã có {len(final_reviews_dict)} Xấu, đang lấy thêm {remaining} Tốt cho đủ {MAX_TOTAL_REVIEWS}...")
                    if await change_sort(page, "Highest"):
                        # Giới hạn giỏ lấy đến MAX_TOTAL_REVIEWS (VD: 40)
                        good_count = await scrape_current_view(page, final_reviews_dict, MAX_TOTAL_REVIEWS)

                print(f"📊 Kết quả: {bad_count} Xấu, {good_count} Tốt. Tổng: {len(final_reviews_dict)}")

                if final_reviews_dict:
                    # 1. Thống kê
                    summary_df = pd.DataFrame([{"restaurant_name": restaurant_name, "bad": bad_count, "good": good_count, "total": len(final_reviews_dict)}])
                    summary_df.to_csv(SUMMARY_FILE, mode='a', index=False, header=not os.path.isfile(SUMMARY_FILE), encoding="utf-8-sig")

                    # 2. Lưu Review và Số sao ra file CSV
                    reviews_list = []
                    for text, star in final_reviews_dict.items():
                        reviews_list.append({
                            "restaurant_name": restaurant_name,
                            "stars": star,
                            "review_text": text
                        })
                    
                    df_reviews = pd.DataFrame(reviews_list)
                    # encoding="utf-8-sig" giúp Excel đọc tiếng Việt không bị lỗi font
                    df_reviews.to_csv(REVIEWS_FILE, mode='a', index=False, header=not os.path.isfile(REVIEWS_FILE), encoding="utf-8-sig")
                    
                    print(f"💾 Đã lưu xong vào {REVIEWS_FILE}.")

            except Exception as e:
                print(f"❌ Lỗi: {e}")
                continue
        await browser.close()

if __name__ == "__main__":
    asyncio.run(run())