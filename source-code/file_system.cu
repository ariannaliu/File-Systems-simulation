#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__device__ __managed__ u32 gtime = 0;


__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
							int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
							int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE, 
							int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS)
{
  // init variables
  fs->volume = volume;

  // init constants
  fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE;
  fs->FCB_SIZE = FCB_SIZE;
  fs->FCB_ENTRIES = FCB_ENTRIES;
  fs->STORAGE_SIZE = VOLUME_SIZE;
  fs->STORAGE_BLOCK_SIZE = STORAGE_BLOCK_SIZE;
  fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;
  fs->MAX_FILE_NUM = MAX_FILE_NUM;
  fs->MAX_FILE_SIZE = MAX_FILE_SIZE;
  fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;
  
  //initial the bits-map
  for(int i=0; i < fs->STORAGE_SIZE; i++){
	  fs->volume[i] = uchar(255);
  }
  
  for(int i=0; i < fs->SUPERBLOCK_SIZE; i++){
	  fs->volume[i] = uchar(0);
  }

}

/*
 * my FCB structure
 * |0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|   20|21   | 22|23 | 24 | 25|26  | 27|28  |
 * |                    file name                    |  location |  size |mode|create_t|modify_t|
 */

__device__ int string_len(char*s) {
	//方程的返回值是包含了\0的数学长度
	int str_len = 0;
	while (true) {
		str_len += 1;
		if (s[str_len] == '\0') {
			break;
		}
	}
	return str_len+1;
}

__device__ int find_first_free_block_offset(FileSystem *fs){
	//函数返回的是 第几个block是free的，offset是从0开始计算的

	int free_bit_map_index;
	int bitmap_num;
	int free_block_offset;
	for(int k=0;k<fs->SUPERBLOCK_SIZE;k++){
		if(fs->volume[k] < 255){
			free_bit_map_index = k;
			bitmap_num = int(fs->volume[k]);
			break;
		}
	}
	//通过bitmap number，确定到底第几个block是空的。应该有八种情况
	if(bitmap_num == 0){
		//0000 0000
		free_block_offset = 8*free_bit_map_index;				
	}else if(bitmap_num == 128){
		//1000 0000
		free_block_offset = 8*free_bit_map_index + 1;
	}else if(bitmap_num == 192){
		//1100 0000
		free_block_offset = 8*free_bit_map_index + 2;
	}else if(bitmap_num == 224){
		//1110 0000
		free_block_offset = 8*free_bit_map_index + 3;
	}else if(bitmap_num == 240){
		//1111 0000
		free_block_offset = 8*free_bit_map_index + 4;
	}else if(bitmap_num == 248){
		//1111 1000
		free_block_offset = 8*free_bit_map_index + 5;
	}else if(bitmap_num == 252){
		//1111 1100
		free_block_offset = 8*free_bit_map_index + 6;
	}else if(bitmap_num == 254){
		//1111 1110
		free_block_offset = 8*free_bit_map_index + 7;
	}
	return free_block_offset;
}

__device__ int trans_num(int a) {
	int result;
	if (a == 0) {
		result = 0;
	}
	else if (a == 128) {
		result = 1;
	}
	else if (a == 192) {
		result = 2;
	}
	else if (a == 224) {
		result = 3;
	}
	else if (a == 240) {
		result = 4;
	}
	else if (a == 248) {
		result = 5;
	}
	else if (a == 252) {
		result = 6;
	}
	else if (a == 254) {
		result = 7;
	}
	else if (a == 255) {
		result = 8;
	}
	return result;
}

__device__ void change_bitmaps(FileSystem *fs, int block_num){
	int bit_exits = 0; 
	int free_bit_map_index;
	
	for (int i = 0; i < fs->SUPERBLOCK_SIZE; i++) {
		
		if (int(fs->volume[i]) == uchar(0)) {
			break;
		}
		
		bit_exits += trans_num(int(fs->volume[i]));
		
	}
	for (int i = 0; i < fs->SUPERBLOCK_SIZE; i++) {
		if (int(fs->volume[i]) < uchar(255)) {
			free_bit_map_index = i;
			//printf("free_bit_map_index is %d\n", free_bit_map_index);
			break;
		}
	}
	//printf("bits_exits is %d\n",bit_exits);
	int exits_byte_num = bit_exits/8;
	int exits_bit_num = bit_exits%8;
	
	int bit_total = bit_exits + block_num;
	int total_byte_num = bit_total/8;
	int total_bit_num = bit_total%8;
	int last_index;
	
	
	
	if(total_byte_num - exits_byte_num == 0){
		last_index = free_bit_map_index;
	}else{
		int distance = total_byte_num-exits_byte_num;
		for(int j=0; j<distance; j++){
			fs->volume[free_bit_map_index + j] = 255;
		}
		last_index = free_bit_map_index+distance;
		
	}
	if(total_bit_num == 1){
		fs->volume[last_index] = 128;
	}else if(total_bit_num == 2){
		fs->volume[last_index] = 192;
	}else if(total_bit_num == 3){
		fs->volume[last_index] = 224;
	}else if(total_bit_num == 4){
		fs->volume[last_index] = 240;
	}else if(total_bit_num == 5){
		fs->volume[last_index] = 248;
	}else if(total_bit_num == 6){
		fs->volume[last_index] = 252;
	}else if(total_bit_num == 7){
		fs->volume[last_index] = 254;
	}
}

__device__ int same_block_size(int orig_size,int size){
	int orig_block = orig_size/32;
	if(orig_size%32>0){
		orig_block += 1;
	}
	
	int curr_block = size/32;
	if(size%32 >0){
		curr_block += 1;
	}
	if(orig_block == curr_block){
		return 1;
	}else if(orig_block < curr_block){
		return 2;
	}else if(orig_block > curr_block){
		return 3;
	}
}

__device__ int sizeToBlock(int size){
	int curr_block = size/32;

	if(size%32 >0){
		curr_block += 1;
	}
	return curr_block;
}


__device__ int find_free_bitmap_index(FileSystem *fs){
	for(int j=0;j<fs->SUPERBLOCK_SIZE;j++){
		if(int(fs->volume[j])<255){
			int free_byte = j;
			return free_byte;
		}
	}
}

__device__ void compact(FileSystem *fs, int block_start_offset, int block_start_index, int orig_size){
	//printf("first bit map is %d\n", int(fs->volume[0]));
	int free_block_offset = find_first_free_block_offset(fs);  //找到第一空的block的offset
	//printf("free block offset is %d\n", free_block_offset);

	int orig_block = sizeToBlock(orig_size); //original size占用多少blocks
	//printf("orig_block is %d\n", orig_block);

	int compact_content_block_num = free_block_offset - block_start_offset - orig_block;
	int orig_byte_size = orig_block*32;
	for(int i=0 ; i< compact_content_block_num*32; i++){
		fs->volume[block_start_index+i] = fs->volume[block_start_index + orig_byte_size+i];
	}
	//printf("here1");
	for(int k=0 ; k<orig_byte_size; k++){
		fs->volume[block_start_index+ compact_content_block_num*32+k] = 255;
	}
	//printf("here2");
	
	//compact bitmap
	int count = orig_block;
	int last_free_byte;
	int last_num;
	int last_block;
	int dis;
	while(count>=0){
		//printf("count!\n");
		last_free_byte = find_free_bitmap_index(fs);
		last_num = int(fs->volume[last_free_byte]);
		//printf("last num is %d\n", last_num);
		if(last_num == 0){
			last_block = 0;
		}else if(last_num = 128){
			last_block = 1;
		}else if(last_num = 192){
			last_block = 2;
		}else if(last_num = 224){
			last_block = 3;
		}else if(last_num = 240){
			last_block = 4;
		}else if(last_num = 248){
			last_block = 5;
		}else if(last_num = 252){
			last_block = 6;
		}else if(last_num = 254){
			last_block = 7;
		}
		//printf("last block is %d\n", last_block);
		if (last_block >= count){       //如果bitmap最后的byte里面存放的blocks足够满足origin的需求
			dis = last_block - count;
			if(dis == 1){
				fs->volume[last_free_byte] = uchar(128);
			}else if(dis == 2){
				fs->volume[last_free_byte] = uchar(192);
			}else if(dis == 3){
				fs->volume[last_free_byte] = uchar(224);
			}else if(dis == 4){
				fs->volume[last_free_byte] = uchar(240);
			}else if(dis == 5){
				fs->volume[last_free_byte] = uchar(248);
			}else if(dis == 6){
				fs->volume[last_free_byte] = uchar(252);
			}else if(dis == 7){
				fs->volume[last_free_byte] = uchar(254);
			}
			break;
		}else{                 //如果bitmap最后的byte不够满足这个需求
			//last_block < count
			fs->volume[last_free_byte] = uchar(0);
			fs->volume[last_free_byte-1] = uchar(254);
			count = count - last_block -1;
		}
	}
}

__device__ int find_last_fcb(FileSystem *fs){
	//return的是最后一个有元素的fcb的index
	int count = -1;
	int last_index;
	for(int i=0; i<1024; i++){
		if(fs->volume[fs->SUPERBLOCK_SIZE+32*i] == uchar(255)){
			count = i;
			break;
		}
	}
	if(count >=0){
		last_index = fs->SUPERBLOCK_SIZE + 32*(count-1);
	}else{
		last_index = fs->FILE_BASE_ADDRESS - 32;     //fcb的最后一个block就是最后一个有元素的index
	}
	return last_index;
}

__device__ int bubble_sort(FileSystem *fs, int op){
	int last_fcb_index = find_last_fcb(fs);          //找到最后一个有元素的fcb的实际index
	int fcb_num = (last_fcb_index - fs->SUPERBLOCK_SIZE)/32 + 1;  //一共有多少个fcb，数学意义上的
	int target1;
	int target2;
	uchar temp[32];
	
	if(op == LS_D){
		target1 = 27;
		target2 = 28;
		for(int i=0; i< fcb_num-1; i++){
			//printf("bubble sort times %d\n",i );
			for(int j=0; j<fcb_num-1-i; j++){
				if(int((fs->volume[fs->SUPERBLOCK_SIZE + 32*j + target1]<<8) + fs->volume[fs->SUPERBLOCK_SIZE + 32*j + target2] ) < int((fs->volume[fs->SUPERBLOCK_SIZE + 32*(j+1) + target1]<<8) + fs->volume[fs->SUPERBLOCK_SIZE + 32*(j+1) + target2] )){
					for (int x=0; x<32; x++){
						temp[x] = fs->volume[fs->SUPERBLOCK_SIZE + 32*j + x];
					}
					for(int y=0; y<32; y++){
						fs->volume[fs->SUPERBLOCK_SIZE + 32*j + y] = fs->volume[fs->SUPERBLOCK_SIZE + 32*(j+1) + y];
					}
					for(int z=0; z<32; z++){
						fs->volume[fs->SUPERBLOCK_SIZE + 32*(j+1) + z] = temp[z];
					}
				}
			}
		}
		
	}else if(op == LS_S){
		target1 = 22;
		target2 = 23;
		for(int i=0; i< fcb_num-1; i++){
			//printf("bubble sort times %d\n", i);
			for(int j=0; j<fcb_num-1-i; j++){
				if(int((fs->volume[fs->SUPERBLOCK_SIZE + 32*j + target1]<<8) + fs->volume[fs->SUPERBLOCK_SIZE + 32*j + target2] ) < int((fs->volume[fs->SUPERBLOCK_SIZE + 32*(j+1) + target1]<<8) + fs->volume[fs->SUPERBLOCK_SIZE + 32*(j+1) + target2] )){
					for (int x=0; x<32; x++){
						temp[x] = fs->volume[fs->SUPERBLOCK_SIZE + 32*j + x];
					}
					for(int y=0; y<32; y++){
						fs->volume[fs->SUPERBLOCK_SIZE + 32*j + y] = fs->volume[fs->SUPERBLOCK_SIZE + 32*(j+1) + y];
					}
					for(int z=0; z<32; z++){
						fs->volume[fs->SUPERBLOCK_SIZE + 32*(j+1) + z] = temp[z];
					}
				}else if(int((fs->volume[fs->SUPERBLOCK_SIZE + 32*j + target1]<<8) + fs->volume[fs->SUPERBLOCK_SIZE + 32*j + target2] ) == int((fs->volume[fs->SUPERBLOCK_SIZE + 32*(j+1) + target1]<<8) + fs->volume[fs->SUPERBLOCK_SIZE + 32*(j+1) + target2] )){
					if(int((fs->volume[fs->SUPERBLOCK_SIZE + 32*j + 25]<<8) + fs->volume[fs->SUPERBLOCK_SIZE + 32*j + 26]) > int((fs->volume[fs->SUPERBLOCK_SIZE + 32*(j+1) + 25]<<8) + fs->volume[fs->SUPERBLOCK_SIZE + 32*(j+1) + 26] )){
						for (int x=0; x<32; x++){
							temp[x] = fs->volume[fs->SUPERBLOCK_SIZE + 32*j + x];
						}
						for(int y=0; y<32; y++){
							fs->volume[fs->SUPERBLOCK_SIZE + 32*j + y] = fs->volume[fs->SUPERBLOCK_SIZE + 32*(j+1) + y];
						}
						for(int z=0; z<32; z++){
							fs->volume[fs->SUPERBLOCK_SIZE + 32*(j+1) + z] = temp[z];
						}
					}
				}
			}
		}
	}
	return fcb_num;
}

__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{	
	//open 函数return的是FCB起始点的实际index
	//printf("This is open function, the length of the file name is %d\n", string_len(s));
	gtime++;
	if (string_len(s)>20){
		printf("ERROR: The length of the file name exceeds the maximum!\n");
		return -1;
	}
	int fcb_index;
	int fcb_find_offset = -3;   //denote the position of the target file in fcb
	int fcb_free_offset = -1;
	int free_bit_map_index=-1;
	int compare_flag = 1;   //1 indicates they are equal, 0 indicates they are different
	int bitmap_num;
	int free_block_offset;
	if(op == G_WRITE){
		//write mode
		for(int i=0; i<1024;i++){             //遍历所有的FCB，找到是否有匹配的名字
			compare_flag = 1;
			fcb_index = fs->SUPERBLOCK_SIZE+i*fs->FCB_SIZE;
			if(fs->volume[fcb_index] == 255){
				//如果没满 但是一定找不到了
				fcb_find_offset = -2;    //did not find file name s in fcb
				fcb_free_offset = i;
				break;
			}
			else{
				//这个不是空的
				for(int j=0; j<string_len(s);j++){
					if(fs->volume[fcb_index + j] != s[j]){
						compare_flag = 0;
						break;
					}
				}
				if(compare_flag == 1){
					//we find the file name
					fcb_find_offset = i;
					break;
				}
			}
		}
		if(fcb_find_offset >=0){
			//find one in fcb
			fcb_index = fs->SUPERBLOCK_SIZE + fcb_find_offset*fs->FCB_SIZE;
			fs->volume[fcb_index + 24] = uchar('w');  //change the mode to write
			return fcb_index;                   //return the actual index of the file information in fcb
		}
		else if(fcb_find_offset == -3){
			printf("ERROR: File system is full, file name does not present!\n");
			return -1;
		}
		else if(fcb_find_offset == -2){
			//fbc没满但是没找到name

			free_block_offset = find_first_free_block_offset(fs);			
			
			fcb_index = fs->SUPERBLOCK_SIZE + fs->FCB_SIZE*fcb_free_offset;
			fs->volume[fcb_index + 24] = uchar('w');
			for (int l=0;l< string_len(s);l++){
				fs->volume[fcb_index + l] = s[l];
			}
			fs->volume[fcb_index + 20] = uchar(free_block_offset >>8);
			fs->volume[fcb_index + 21] = uchar(free_block_offset & 0x000000FF);
			fs->volume[fcb_index + 25] = uchar(gtime >>8);
			fs->volume[fcb_index + 26] = uchar(gtime & 0x000000FF);
			return fcb_index;
		}
	}
	else if(op == G_READ){
		//read mode
		for(int i=0; i<1024;i++){
			compare_flag = 1;
			fcb_index = fs->SUPERBLOCK_SIZE+i*fs->FCB_SIZE;
			if(fs->volume[fcb_index] == 255){
				fcb_find_offset = -2;    //did not find file name s in fcb
				break;
			}
			else{
				for(int j=0; j< string_len(s);j++){
					if(fs->volume[fcb_index + j] != s[j]){
						compare_flag = 0;
						break;
					}
				}
				if(compare_flag == 1){
					fcb_find_offset = i;
					break;
				}
			}
		}
		if(fcb_find_offset == -3 || fcb_find_offset==-2){
			printf("ERROR: The file is not in the file system, cannot read it \n");
			return -1;
		}else{
			//printf("fcb_find_offset is %d\n", fcb_find_offset);
			fcb_index = fs->SUPERBLOCK_SIZE+fs->FCB_SIZE*fcb_find_offset;
			//printf("fcb_index is %d\n", fcb_index);
			fs->volume[fcb_index+24] = uchar('r');
			return fcb_index;
		}
	}
}


__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp)
{
	if(fs->volume[fp+24]== uchar('w')){
		printf("ERROR: Cannnot implement read on write file!\n");
		return;
	}
	//printf("fcb_index is %d\n", fp);
	int block_offset = int((fs->volume[fp+20]<<8) + fs->volume[fp+21]);
	//printf("block_offset is %d\n", block_offset);
	int block_addr = fs->FILE_BASE_ADDRESS + fs->STORAGE_BLOCK_SIZE *block_offset;
	//printf("block_addr is %d\n", block_addr);
	
	for(int i=0; i<size; i++){
		output[i] = fs->volume[block_addr + i];
	}
}

__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp)
{
	/* Implement write operation here */
	gtime++;
	
	if(fs->volume[fp+24] == uchar('r')){
		printf("ERROR: Cannnot implement write on read mode file!\n");
		return -1;
	}
	if(size>1024){
		printf("ERROR: File size exceeds the maximum!\n");
		return -1;
	}
	
	int free_block_offset;
	int free_block_index;
	int block_num = size/32;         //存这个size需要多少的blocks
	int free_bit_map_index;
	if(size%32 != 0){
		block_num += 1;
	}
	if(fs->volume[fp+22]==255){
		//里面还没有存过东西
		//判断还够不够放
		//printf("here!!!\n");
		free_block_offset = find_first_free_block_offset(fs);
		if(fs->volume[fs->SUPERBLOCK_SIZE- (block_num+free_block_offset)*fs->STORAGE_BLOCK_SIZE] != 0){
			printf("ERROR: Space is not enough to write\n");
			return -1;
		}
		//printf("bitsss 1 = %d\n", int(fs->volume[0]));
		//够放置
		//改写bitmap往后面多写，意思是把0替换成1，不用考虑把1换成0的情况
		//printf("block_num 1 = %d\n", block_num);
		change_bitmaps(fs, block_num);
		
		//printf("bitsss 1 = %d\n", int(fs->volume[0]));
		free_block_offset = int((fs->volume[fp+20]<<8)+(fs->volume[fp+21]));
		//printf("free_block_offset is %d\n", free_block_offset);
		free_block_index = fs->FILE_BASE_ADDRESS+ free_block_offset*32;   //第一个空的block的index
		for(int i=0; i<size; i++){
			fs->volume[free_block_index + i] = input[i]; 
		}
		fs->volume[fp+22] = uchar(size>>8);
		fs->volume[fp+23] = uchar(size & 0x000000FF);
		fs->volume[fp + 27] = uchar(gtime >>8);
		fs->volume[fp + 28] = uchar(gtime & 0x000000FF);
		
	}else{
		//里面存过东西了
		//printf("GEEEEEEEEEEEEET IN HERE\n");
		int orig_size = int(((fs->volume[fp+22])<<8)+(fs->volume[fp+23]));
		int block_flag = same_block_size(orig_size,size);
		int block_start_offset = int((fs->volume[fp+20]<<8) + fs->volume[fp+21]);
		int block_start_index = fs->FILE_BASE_ADDRESS + block_start_offset*fs->STORAGE_BLOCK_SIZE;
		//把原来位置上的东西都清掉
		for (int j=0 ; j<orig_size; j++){
			fs->volume[block_start_index + j] = 255;
		}
		//printf("DELETE DONE!\n");
		if(block_flag == 1){
			//printf("YEAH, NO COMPACT!!\n");
			//位置没变，不用compact
			for(int n=0;n<size;n++){
				fs->volume[block_start_index + n] = input[n];
			}			
		}else if(block_flag == 2 || block_flag == 3){
			//printf("ON NO, WE need to compact!\n");
			compact(fs, block_start_offset, block_start_index, orig_size);
			//printf("DID I COME COUT?\n");
			int new_free_block_offset = find_first_free_block_offset(fs);
			
			if(block_flag == 2){
				//orig<curr
				if(size > (fs->STORAGE_SIZE-(new_free_block_offset*32+fs->FILE_BASE_ADDRESS))){
					printf("ERROR: No space for this writern file!\n");
					return -1;
				}	
			}
			fs->volume[fp+20] = uchar(new_free_block_offset>>8);
			fs->volume[fp+21] = uchar(new_free_block_offset & 0x000000FF);
			
			//write
			change_bitmaps(fs, block_num);
			free_block_offset = find_first_free_block_offset(fs);
				free_block_index = fs->FILE_BASE_ADDRESS + free_block_offset*32;
			for(int i=0; i<size; i++){
				fs->volume[free_block_index + i] = input[i]; 
			}
		}
		fs->volume[fp+22] = uchar(size>>8);
		fs->volume[fp+23] = uchar(size & 0x000000FF);
		fs->volume[fp + 27] = uchar(gtime >>8);
		fs->volume[fp + 28] = uchar(gtime & 0x000000FF);
	}
	//printf("\nAfter write function AAAAAAAAAAAAAAAAAA: size is %d\n", int((fs->volume[fp + 22] << 8) + fs->volume[fp + 23]));
	
}
__device__ void fs_gsys(FileSystem *fs, int op)
{
	/* Implement LS_D and LS_S operation here */
	int file_num = bubble_sort(fs,op);
	//printf("filenum = %d\n",file_num);
	int fcb_addr;
	uchar target;
	if(op == LS_D){
		printf("===sort by modified time===\n");
		for (int i=0; i<file_num; i++){
			//printf("here\n");
			fcb_addr = fs->SUPERBLOCK_SIZE + 32*i;
			//printf("fcb_addr is %d\n", fcb_addr);
			for(int j=0; j<20; j++){
				if(int(fs->volume[fcb_addr + j]) == int('\0')){
					printf("\n");
					break;
				}
				printf("%c",uchar(fs->volume[fcb_addr + j]));
			}
		}
	}else if(op == LS_S){
		printf("===sort by file size===\n");
		for (int i=0; i<file_num; i++){
			fcb_addr = fs->SUPERBLOCK_SIZE + 32*i;
			for(int j=0; j<20; j++){
				if(int(fs->volume[fcb_addr + j]) == int('\0')){
					printf(" %d\n",int((fs->volume[fcb_addr+22]<<8) + fs->volume[fcb_addr+23]));
					break;
				}
				printf("%c",uchar(fs->volume[fcb_addr + j]));
			}
		}
	}
}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
	if(string_len(s)>20){
		printf("ERROR: file name exceeds the maximum!\n");
	}
	/* Implement rm operation here */
	int fcb_index;
	int fcb_find_offset = -3;   //denote the position of the target file in fcb
	int compare_flag = 1;   //1 indicates they are equal, 0 indicates they are different
	if(op == RM){
		//find the file name
		for(int i=0; i<1024;i++){
			compare_flag = 1;
			fcb_index = fs->SUPERBLOCK_SIZE+i*32;
			if(fs->volume[fcb_index] == 255){
				//如果没满 但是一定找不到了
				fcb_find_offset = -2;    //did not find file name s in fcb
				break;
			}else{
				//这个不是空的
				for(int j=0; j< string_len(s);j++){
					if(fs->volume[fcb_index + j] != s[j]){
						compare_flag = 0;
						break;
					}
				}
				if(compare_flag == 1){
					//we find the file name
					fcb_find_offset = i;
					break;
				}
			}
		}
		if(fcb_find_offset==-3 || fcb_find_offset==-2){
			printf("ERROR: Cannnot find the file name in file system!\n");
			return;
		}else if(fcb_find_offset>=0){
			//find file name in fcb
			
			//操作file memory+bitmap
			int fcb_addr = fs->SUPERBLOCK_SIZE + 32*fcb_find_offset;
			int block_start_offset = int((fs->volume[fcb_addr+20]<<8) + fs->volume[fcb_addr+21]);
			int block_start_index = fs->FILE_BASE_ADDRESS + 32*block_start_offset;
			int size = int((fs->volume[fcb_addr+22]<<8) + fs->volume[fcb_addr+23]);
			for(int j=0; j<size; j++){
				fs->volume[block_start_index+j] = 255;
			}
			compact(fs, block_start_offset, block_start_index, size);
			
			
			//操作FCB
			int last_fcb_start_index = find_last_fcb(fs);
			
			for(int y=0; y<32; y++){
				fs->volume[fcb_addr+y] = fs->volume[last_fcb_start_index+y];
			}
			for(int k=0; k<32; k++){
				fs->volume[last_fcb_start_index+k] = 255;
			}
			
		}
		
		
	}
	else{
		printf("ERROR: Invalid instruction!\n");
		return;
	}
}
